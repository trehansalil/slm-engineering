"""Local SFT fine-tuning on Mac (MPS) or CPU.

Continues training the SFT model locally using the tokenized dataset
from Modal. Downloads tokenized data from the volume if not present.
Resumes from the latest local checkpoint automatically.

Usage:
    python sft/finetune_local.py                        # 3 epochs, auto device
    python sft/finetune_local.py --n-epochs 10          # more epochs
    python sft/finetune_local.py --lr 1e-5              # custom learning rate
    python sft/finetune_local.py --device cpu           # force CPU
    python sft/finetune_local.py --resume local_model/sft_local_final

Requires: pip install torch transformers numpy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, LlamaForCausalLM

MODEL_DIR = "local_model/base"
TOKENS_DIR = "local_model/tokens"
METRICS_PATH = "local_model/sft_local_metrics.jsonl"
CKPT_DIR = "local_model/checkpoints"
OUTPUT_DIR = "local_model/sft_local"
VOLUME_NAME = "slm-125m"

IGNORE_INDEX = -100


def _tokenize_chat(messages, tokenizer):
    """Tokenize a chat conversation, returning input_ids and labels."""
    role_map = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    eos_id = tokenizer.convert_tokens_to_ids("<|eos|>")
    input_ids = []
    labels = []
    for msg in messages:
        role_id = tokenizer.convert_tokens_to_ids(role_map[msg["role"]])
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        turn_ids = [role_id] + content_ids + [eos_id]
        if msg["role"] == "assistant":
            turn_labels = [IGNORE_INDEX] + content_ids + [eos_id]
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)
        input_ids.extend(turn_ids)
        labels.extend(turn_labels)
    return input_ids, labels


def tokenize_locally(tokenizer, seq_len=1024):
    """Tokenize SFT data locally using the base model's tokenizer."""
    import json as _json
    sft_path = "data/sft_train.jsonl"
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")

    with open(sft_path, encoding="utf-8") as fh:
        examples = [_json.loads(line) for line in fh]
    print(f"Tokenizing {len(examples)} examples locally (seq_len={seq_len})...")

    all_input_ids, all_labels = [], []
    skipped_short = 0
    for ex in examples:
        input_ids, labels = _tokenize_chat(ex["messages"], tokenizer)
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
        if len(input_ids) < 20:
            skipped_short += 1
            continue
        pad_len = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    if not all_input_ids:
        raise ValueError(f"No valid examples after tokenization ({len(examples)} input)")

    os.makedirs(TOKENS_DIR, exist_ok=True)
    np.save(f"{TOKENS_DIR}/input_ids.npy", np.array(all_input_ids, dtype=np.int32))
    np.save(f"{TOKENS_DIR}/labels.npy", np.array(all_labels, dtype=np.int32))
    print(f"Tokenized {len(all_input_ids)} examples (skipped {skipped_short} short)")


def download_tokens(tokenizer):
    """Tokenize locally if .npy files missing, or download from Modal."""
    ids_path = f"{TOKENS_DIR}/input_ids.npy"
    if os.path.exists(ids_path):
        print("Tokenized data ready")
        return
    sft_path = "data/sft_train.jsonl"
    if os.path.exists(sft_path):
        tokenize_locally(tokenizer)
    else:
        os.makedirs(TOKENS_DIR, exist_ok=True)
        for fname in ("input_ids.npy", "labels.npy"):
            local_path = f"{TOKENS_DIR}/{fname}"
            print(f"Downloading {fname} from Modal volume...")
            subprocess.run(
                ["modal", "volume", "get", VOLUME_NAME,
                 f"sft/tokens/{fname}", local_path],
                check=True,
            )
        print("Tokenized data ready")


class SFTDataset(Dataset):
    def __init__(self, input_ids, labels, pad_id=0):
        self.examples = []
        for ids_row, lab_row in zip(input_ids, labels):
            mask = ids_row != pad_id
            length = int(mask.sum())
            if length < 2:
                continue
            self.examples.append((
                ids_row[:length].astype(np.int64),
                lab_row[:length].astype(np.int64),
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        ids, labs = self.examples[i]
        return torch.from_numpy(ids), torch.from_numpy(labs)


def collate_fn(batch, pad_id=2):
    """Dynamic padding — pad to longest in batch, not to 1024."""
    ids_list, _ = zip(*batch)
    max_len = max(x.size(0) for x in ids_list)
    padded_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    padded_labs = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, (ids, labs) in enumerate(batch):
        padded_ids[i, :ids.size(0)] = ids
        padded_labs[i, :labs.size(0)] = labs
        attn_mask[i, :ids.size(0)] = 1
    return padded_ids, padded_labs, attn_mask


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, val_dl, device):
    model.eval()
    total_loss = total_count = 0
    with torch.no_grad():
        for ids, labs, mask in val_dl:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss
            total_loss += loss.item() * ids.size(0)
            total_count += ids.size(0)
    model.train()
    return total_loss / total_count


def find_latest_checkpoint():
    """Find the latest checkpoint in CKPT_DIR by epoch number."""
    if not os.path.exists(CKPT_DIR):
        return None, 0
    ckpts = [d for d in os.listdir(CKPT_DIR) if d.startswith("epoch_")]
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: int(x.split("_")[1]))
    latest = ckpts[-1]
    epoch_num = int(latest.split("_")[1])
    return f"{CKPT_DIR}/{latest}", epoch_num


def save_model_bf16(model, path):
    """Save model weights in bfloat16 to halve disk usage."""
    original_dtype = next(model.parameters()).dtype
    model.to(torch.bfloat16)
    model.save_pretrained(path)
    model.to(original_dtype)


def save_checkpoint(model, optimizer, epoch, step, val_loss, ppl, tokenizer_src):
    """Save a checkpoint with model, optimizer state, and tokenizer."""
    ckpt_path = f"{CKPT_DIR}/epoch_{epoch:02d}"
    os.makedirs(ckpt_path, exist_ok=True)
    save_model_bf16(model, ckpt_path)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "ppl": ppl,
    }, f"{ckpt_path}/training_state.pt")
    for tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = f"{tokenizer_src}/{tf}"
        dst = f"{ckpt_path}/{tf}"
        if os.path.exists(src):
            shutil.copy2(src, dst)
    print(f"  [ckpt] saved epoch {epoch} to {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="Local SFT fine-tuning")
    parser.add_argument("--model", default=MODEL_DIR, help="Base model directory")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from specific checkpoint dir")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing checkpoints, start from base model")
    parser.add_argument("--n-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every-epochs", type=int, default=2,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--retokenize", action="store_true",
                        help="Force re-tokenization of SFT data")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.convert_tokens_to_ids("<|pad|>")

    if args.retokenize:
        if os.path.exists(TOKENS_DIR):
            shutil.rmtree(TOKENS_DIR)
            print("Cleared old tokenized data")

    download_tokens(tokenizer)

    input_ids = np.load(f"{TOKENS_DIR}/input_ids.npy")
    labels = np.load(f"{TOKENS_DIR}/labels.npy")
    print(f"Loaded {len(input_ids)} tokenized examples")

    full_ds = SFTDataset(input_ids, labels, pad_id=pad_id)
    val_size = max(100, int(len(full_ds) * 0.05))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # Determine where to load model from (resume or fresh)
    start_epoch = 0
    resume_path = args.resume

    if resume_path is None and not args.fresh:
        latest_ckpt, latest_epoch = find_latest_checkpoint()
        if latest_ckpt:
            resume_path = latest_ckpt
            start_epoch = latest_epoch

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        model = LlamaForCausalLM.from_pretrained(resume_path, torch_dtype=torch.float32)
        state_path = f"{resume_path}/training_state.pt"
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            start_epoch = state["epoch"]
            print(f"  Resuming after epoch {start_epoch} "
                  f"(val_loss={state.get('val_loss', '?')}, ppl={state.get('ppl', '?')})")
    else:
        print(f"Loading base model from {args.model}...")
        model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)

    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = pad_id

    model = model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e6:.1f}M params")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=False, drop_last=True,
                          collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                        num_workers=0, pin_memory=False,
                        collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Restore optimizer state if available
    if resume_path and os.path.exists(f"{resume_path}/training_state.pt"):
        opt_state = torch.load(f"{resume_path}/training_state.pt",
                               map_location="cpu", weights_only=False)
        if "optimizer" in opt_state:
            optimizer.load_state_dict(opt_state["optimizer"])
            print("  Restored optimizer state")

    total_epochs = args.n_epochs
    remaining_epochs = total_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Already trained {start_epoch} epochs (target {total_epochs}). Nothing to do.")
        return

    steps_per_epoch = len(train_dl) // args.grad_accum
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return args.lr * step / max(1, warmup_steps)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * ratio))

    print(f"\nTraining: epochs {start_epoch+1} -> {total_epochs}, "
          f"{total_steps} steps, batch={args.batch_size}x{args.grad_accum}, lr={args.lr}")
    print(f"Checkpoint every {args.ckpt_every_epochs} epochs")
    print(f"Estimated time: ~{total_steps * 1.0 / 60:.0f} min\n")

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    mf = open(METRICS_PATH, "a")

    # Perplexity log header
    ppl_log_path = "local_model/perplexity_log.jsonl"
    ppl_log = open(ppl_log_path, "a")

    # Initial eval
    init_vl = evaluate(model, val_dl, device)
    init_ppl = math.exp(min(init_vl, 20))
    print(f"Starting val_loss={init_vl:.4f} ppl={init_ppl:.2f}")
    ppl_log.write(json.dumps({
        "epoch": start_epoch, "val_loss": round(init_vl, 4),
        "ppl": round(init_ppl, 2), "phase": "start",
    }) + "\n")
    ppl_log.flush()

    step = 0
    micro = 0
    running_loss = 0.0
    best_val_loss = init_vl
    t0 = t_global = time.time()

    # Restore step from checkpoint
    if resume_path and os.path.exists(f"{resume_path}/training_state.pt"):
        ckpt_state = torch.load(f"{resume_path}/training_state.pt",
                                map_location="cpu", weights_only=False)
        step = ckpt_state.get("step", 0)

    for epoch_idx in range(remaining_epochs):
        current_epoch = start_epoch + epoch_idx + 1
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        for ids, labs, mask in train_dl:
            ids, labs, mask = ids.to(device), labs.to(device), mask.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labs).loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()
            epoch_train_loss += loss.item()
            micro += 1

            if micro % args.grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1
            epoch_train_steps += 1

            if step % args.log_every == 0:
                avg = running_loss / args.log_every
                vl_step = evaluate(model, val_dl, device)
                elapsed = time.time() - t_global
                sec_per_step = (time.time() - t0) / args.log_every
                print(f"step {step:>5}/{total_steps} | train_loss {avg:.4f} | "
                      f"val_loss {vl_step:.4f} | "
                      f"lr {current_lr:.2e} | {sec_per_step:.2f}s/step | "
                      f"{elapsed/60:.1f}min")
                mf.write(json.dumps({"epoch": current_epoch, "step": step,
                                     "train_loss": round(avg, 4),
                                     "val_loss": round(vl_step, 4),
                                     "lr": current_lr}) + "\n")
                mf.flush()
                running_loss = 0.0
                t0 = time.time()

        # End of epoch — always evaluate
        avg_train_loss = epoch_train_loss / max(1, epoch_train_steps)
        vl = evaluate(model, val_dl, device)
        ppl = math.exp(min(vl, 20))
        elapsed = time.time() - t_global
        improving = "✓" if vl < best_val_loss else "✗"
        print(f"  [epoch {current_epoch}] train_loss={avg_train_loss:.4f} "
              f"val_loss={vl:.4f} ppl={ppl:.2f} "
              f"{improving} | {elapsed/60:.1f}min")

        ppl_log.write(json.dumps({
            "epoch": current_epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(vl, 4),
            "ppl": round(ppl, 2), "improving": vl < best_val_loss,
        }) + "\n")
        ppl_log.flush()

        saved_ckpt = False
        if vl < best_val_loss:
            best_val_loss = vl
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            save_model_bf16(model, OUTPUT_DIR)
            for tf in ("tokenizer.json", "tokenizer_config.json",
                       "special_tokens_map.json"):
                src = f"{args.model}/{tf}"
                dst = f"{OUTPUT_DIR}/{tf}"
                if os.path.exists(src):
                    shutil.copy2(src, dst)
            save_checkpoint(model, optimizer, current_epoch, step, vl, ppl, args.model)
            saved_ckpt = True
            print(f"  [best] saved to {OUTPUT_DIR} + checkpoint")

        # Checkpoint every N epochs (skip if already saved this epoch)
        if current_epoch % args.ckpt_every_epochs == 0 and not saved_ckpt:
            save_checkpoint(model, optimizer, current_epoch, step, vl, ppl, args.model)

    # Final save
    final_dir = f"{OUTPUT_DIR}_final"
    os.makedirs(final_dir, exist_ok=True)
    save_model_bf16(model, final_dir)
    for tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = f"{args.model}/{tf}"
        dst = f"{final_dir}/{tf}"
        if os.path.exists(src):
            shutil.copy2(src, dst)

    final_vl = evaluate(model, val_dl, device)
    final_ppl = math.exp(min(final_vl, 20))
    ppl_log.write(json.dumps({
        "epoch": total_epochs, "val_loss": round(final_vl, 4),
        "ppl": round(final_ppl, 2), "phase": "final",
    }) + "\n")
    mf.close()
    ppl_log.close()
    elapsed = time.time() - t_global

    print(f"\nDone! {step} steps in {elapsed/60:.1f}min")
    print(f"Epochs: {start_epoch+1} -> {total_epochs}")
    print(f"Final val_loss={final_vl:.4f} ppl={final_ppl:.2f}")
    print(f"Best val_loss={best_val_loss:.4f} ppl={math.exp(min(best_val_loss, 20)):.2f}")
    print(f"\nCheckpoints: {CKPT_DIR}/")
    print(f"Best model: {OUTPUT_DIR}/")
    print(f"Perplexity log: {ppl_log_path}")
    print(f"\nTo chat: python inference/chat_pytorch.py --model {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
