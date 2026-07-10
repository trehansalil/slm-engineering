"""DDP training script for 125M SLM. Launched via torchrun from Modal."""

from __future__ import annotations

import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


class WindowDataset(Dataset):
    def __init__(self, bin_dir: str, seq_len: int):
        files = sorted(glob.glob(f"{bin_dir}/*.bin"))
        arrays = []
        for f in files:
            raw = np.fromfile(f, dtype=np.uint16)
            n = len(raw) // seq_len
            if n > 0:
                arrays.append(raw[: n * seq_len].reshape(n, seq_len))
        self.data = np.concatenate(arrays)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return torch.from_numpy(self.data[i].astype(np.int64))


def get_lr(step: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if step < warmup:
        return lr_max * step / warmup
    ratio = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * ratio))


def evaluate(model, val_dl, device):
    model.eval()
    total_loss = total_count = 0
    with torch.no_grad():
        for batch in val_dl:
            x = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss
            total_loss += loss.item() * x.size(0)
            total_count += x.size(0)
    t = torch.tensor([total_loss, total_count], device=device, dtype=torch.float64)
    dist.all_reduce(t)
    model.train()
    return (t[0] / t[1]).item()


def main():
    import config
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = rank == 0

    torch.manual_seed(config.TRAIN.seed + rank)

    # Model
    model = LlamaForCausalLM(LlamaConfig(**config.MODEL.to_llama_kwargs()))
    model = model.to(device=device, dtype=torch.bfloat16)
    if is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"Model: {n:,} params ({n/1e6:.1f}M)")
    model = DDP(model, device_ids=[local_rank])

    # Data
    train_ds = WindowDataset(config.TRAIN_TOKENS_DIR, config.SEQ_LEN)
    val_ds = WindowDataset(config.VAL_TOKENS_DIR, config.SEQ_LEN)
    if is_main:
        print(f"Train: {len(train_ds):,} windows = {len(train_ds)*config.SEQ_LEN/1e9:.2f}B tok")
        print(f"Val:   {len(val_ds):,} windows = {len(val_ds)*config.SEQ_LEN/1e6:.1f}M tok")

    tc = config.TRAIN
    per_gpu_batch = tc.global_batch_tokens // (config.SEQ_LEN * world_size)
    grad_accum = max(1, per_gpu_batch // tc.micro_batch_size)

    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=tc.seed)
    train_dl = DataLoader(train_ds, batch_size=tc.micro_batch_size, sampler=sampler,
                          num_workers=4, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=tc.micro_batch_size * 4, shuffle=False,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                                  betas=(tc.beta1, tc.beta2),
                                  weight_decay=tc.weight_decay)

    n_epochs = int(os.environ.get("NUM_EPOCHS", "3"))
    total_steps = (len(train_ds) * config.SEQ_LEN * n_epochs) // tc.global_batch_tokens
    warmup_steps = tc.warmup_tokens // tc.global_batch_tokens

    # Resume
    start_step = 0
    if os.path.exists(config.RESUME_CKPT_PATH):
        ckpt = torch.load(config.RESUME_CKPT_PATH, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if is_main:
            print(f"Resumed from step {start_step}")

    if is_main:
        print(f"Config: {n_epochs} epochs, {total_steps} steps, warmup {warmup_steps}, "
              f"grad_accum {grad_accum}, world_size {world_size}")
        os.makedirs(os.path.dirname(config.METRICS_PATH), exist_ok=True)
        mf = open(config.METRICS_PATH, "a")

    step = start_step
    micro = 0
    running_loss = 0.0
    t0 = t_global = time.time()

    for epoch in range(n_epochs):
        if step >= total_steps:
            break
        sampler.set_epoch(epoch)
        for batch in train_dl:
            if step >= total_steps:
                break

            x = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=x, labels=x).loss / grad_accum
            loss.backward()
            running_loss += loss.item()
            micro += 1

            if micro % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            lr = get_lr(step, warmup_steps, total_steps, tc.lr, tc.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if is_main and step % tc.log_every_steps == 0:
                dt = time.time() - t0
                tps = tc.global_batch_tokens * tc.log_every_steps / dt
                avg = running_loss / tc.log_every_steps
                elapsed = time.time() - t_global
                print(f"step {step:>6}/{total_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                      f"{tps/1e6:.2f}M tok/s | {elapsed/60:.1f}min")
                mf.write(json.dumps({"step": step, "loss": round(avg, 4), "lr": lr,
                                     "tok_per_sec": round(tps),
                                     "elapsed_s": round(elapsed)}) + "\n")
                mf.flush()
                running_loss = 0.0
                t0 = time.time()

            if step % tc.eval_every_steps == 0:
                vl = evaluate(model, val_dl, device)
                if is_main:
                    print(f"  [eval] val_loss={vl:.4f} ppl={math.exp(min(vl, 20)):.2f}")
                dist.barrier()

            if step % tc.ckpt_every_steps == 0 and is_main:
                os.makedirs(config.CKPT_DIR, exist_ok=True)
                torch.save({"step": step, "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict()}, config.RESUME_CKPT_PATH)
                print(f"  [ckpt] step {step}")

        if is_main:
            print(f"  [epoch {epoch+1}/{n_epochs} done]")

    # Final save
    if is_main:
        os.makedirs(config.BASE_CKPT_DIR, exist_ok=True)
        model.module.save_pretrained(config.BASE_CKPT_DIR)
        tok = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
        tok.save_pretrained(config.BASE_CKPT_DIR)
        elapsed = time.time() - t_global
        print(f"Done! {step} steps, {step * tc.global_batch_tokens / 1e9:.2f}B tokens seen, "
              f"{elapsed/60:.1f}min")
        mf.close()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
