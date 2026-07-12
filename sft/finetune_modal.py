"""SFT fine-tuning for the 125M SLM on Modal.

Tokenizes chat JSONL, masks loss on system/user turns, fine-tunes from
the pre-trained base checkpoint with LoRA-free full fine-tuning.

Usage:
    modal run sft/finetune_modal.py::tokenize_sft
    modal run sft/finetune_modal.py::finetune
    modal run sft/finetune_modal.py::finetune --n-epochs 5
"""

from __future__ import annotations

import modal

import config

app = modal.App("slm125m-sft")

volume = modal.Volume.from_name(config.VOLUME_NAME, create_if_missing=True)
VOLUMES = {config.DATA_ROOT: volume}

SFT_DIR = f"{config.DATA_ROOT}/sft"
SFT_TOKENS_DIR = f"{SFT_DIR}/tokens"
BASE_CKPT_DIR = f"{config.DATA_ROOT}/checkpoints_10epoch"
SFT_CKPT_DIR = f"{config.DATA_ROOT}/checkpoints_sft"
SFT_METRICS_PATH = f"{SFT_CKPT_DIR}/sft_metrics.jsonl"
SFT_TRAIN_PATH = f"{SFT_DIR}/sft_train.jsonl"

HF_BASE_REPO = "thesreedath/slm-125m-base"

IGNORE_INDEX = -100

sft_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers==4.46.3",
        "numpy",
        "huggingface_hub",
    )
    .add_local_python_source("config")
)


def _tokenize_chat(messages: list[dict], tokenizer) -> tuple[list[int], list[int]]:
    """Tokenize a chat conversation, returning input_ids and labels.

    Labels are IGNORE_INDEX for system/user tokens — only assistant
    responses contribute to the loss.
    """
    role_map = {
        "system": "<|system|>",
        "user": "<|user|>",
        "assistant": "<|assistant|>",
    }
    eos_id = tokenizer.convert_tokens_to_ids(config.SPECIAL_TOKENS["eos_token"])

    input_ids = []
    labels = []

    for msg in messages:
        role_token = role_map[msg["role"]]
        role_id = tokenizer.convert_tokens_to_ids(role_token)
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        turn_ids = [role_id] + content_ids + [eos_id]

        if msg["role"] == "assistant":
            turn_labels = [IGNORE_INDEX] + content_ids + [eos_id]
        else:
            turn_labels = [IGNORE_INDEX] * len(turn_ids)

        input_ids.extend(turn_ids)
        labels.extend(turn_labels)

    return input_ids, labels


@app.function(
    image=sft_image,
    volumes=VOLUMES,
    timeout=60 * 15,
    cpu=4.0,
    memory=8_192,
)
def tokenize_sft_data() -> dict:
    """Tokenize the SFT chat JSONL into padded numpy arrays."""
    import json
    import os

    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_DIR)
    pad_id = tokenizer.convert_tokens_to_ids(config.SPECIAL_TOKENS["pad_token"])
    seq_len = config.SEQ_LEN

    with open(SFT_TRAIN_PATH, encoding="utf-8") as fh:
        examples = [json.loads(line) for line in fh]

    print(f"Tokenizing {len(examples)} examples (seq_len={seq_len})...")

    all_input_ids = []
    all_labels = []
    skipped_long = 0
    skipped_short = 0
    lengths = []

    for ex in examples:
        input_ids, labels = _tokenize_chat(ex["messages"], tokenizer)
        if len(input_ids) > seq_len:
            input_ids = input_ids[:seq_len]
            labels = labels[:seq_len]
            skipped_long += 1
        if len(input_ids) < 20:
            skipped_short += 1
            continue

        lengths.append(len(input_ids))
        pad_len = seq_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [IGNORE_INDEX] * pad_len

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    input_ids_arr = np.array(all_input_ids, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int32)

    os.makedirs(SFT_TOKENS_DIR, exist_ok=True)
    np.save(f"{SFT_TOKENS_DIR}/input_ids.npy", input_ids_arr)
    np.save(f"{SFT_TOKENS_DIR}/labels.npy", labels_arr)
    volume.commit()

    lengths_arr = np.array(lengths)
    stats = {
        "total_examples": len(examples),
        "tokenized": len(all_input_ids),
        "truncated": skipped_long,
        "skipped_short": skipped_short,
        "avg_length": int(lengths_arr.mean()),
        "median_length": int(np.median(lengths_arr)),
        "p95_length": int(np.percentile(lengths_arr, 95)),
        "max_length": int(lengths_arr.max()),
    }
    print(f"Tokenization complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


@app.local_entrypoint()
def tokenize_sft():
    """Tokenize the SFT dataset."""
    stats = tokenize_sft_data.remote()
    print(f"Done: {stats}")


@app.function(
    image=sft_image,
    volumes=VOLUMES,
    gpu="A100",
    timeout=3600 * 2,
)
def run_sft(n_epochs: int = 3) -> dict:
    """Fine-tune the pre-trained base model on the SFT dataset."""
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
    from transformers import LlamaConfig, LlamaForCausalLM

    device = torch.device("cuda")

    # Load tokenized SFT data
    input_ids = np.load(f"{SFT_TOKENS_DIR}/input_ids.npy")
    labels = np.load(f"{SFT_TOKENS_DIR}/labels.npy")
    print(f"Loaded {len(input_ids)} tokenized examples")

    class SFTDataset(Dataset):
        def __init__(self, input_ids, labels):
            self.input_ids = input_ids
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, i):
            return (
                torch.from_numpy(self.input_ids[i].astype(np.int64)),
                torch.from_numpy(self.labels[i].astype(np.int64)),
            )

    full_ds = SFTDataset(input_ids, labels)
    val_size = max(100, int(len(full_ds) * 0.05))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    # SFT hyperparams
    lr = 2e-5
    min_lr = 2e-6
    batch_size = 16
    grad_accum = 2
    weight_decay = 0.01
    grad_clip = 1.0
    log_every = 20
    eval_every = 200
    ckpt_every = 500

    # Model — resume from SFT checkpoint if available, otherwise from base
    from huggingface_hub import snapshot_download

    sft_ckpt_path = f"{SFT_CKPT_DIR}/sft_ckpt.pt"
    start_step = 0
    prior_epochs = 0

    if os.path.exists(sft_ckpt_path):
        print("Resuming from SFT checkpoint...")
        model = LlamaForCausalLM.from_pretrained(BASE_CKPT_DIR, torch_dtype=torch.bfloat16)
        model = model.to(device=device)
        ckpt = torch.load(sft_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        start_step = ckpt["step"]
        steps_per_epoch = len(DataLoader(train_ds, batch_size=batch_size, drop_last=True)) // grad_accum
        prior_epochs = start_step // max(1, steps_per_epoch)
        print(f"Resumed from step {start_step} (~{prior_epochs} prior epochs)")
    else:
        if not os.path.exists(f"{BASE_CKPT_DIR}/config.json"):
            print(f"Downloading base model from {HF_BASE_REPO}...")
            snapshot_download(
                repo_id=HF_BASE_REPO,
                local_dir=BASE_CKPT_DIR,
                ignore_patterns=["*.md", ".gitattributes"],
            )
            volume.commit()
            print("Base model downloaded and cached on volume")

        model = LlamaForCausalLM.from_pretrained(BASE_CKPT_DIR, torch_dtype=torch.bfloat16)
        print(f"Loaded pre-trained model from {HF_BASE_REPO}")
        model = model.to(device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
    )
    if start_step > 0 and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Restored optimizer state")

    steps_per_epoch = len(train_dl) // grad_accum
    total_steps = start_step + steps_per_epoch * n_epochs
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

    print(f"SFT config: {n_epochs} epochs (+ {prior_epochs} prior), {total_steps} steps, "
          f"batch={batch_size}x{grad_accum}, lr={lr}, warmup={warmup_steps}")

    os.makedirs(SFT_CKPT_DIR, exist_ok=True)
    mf = open(SFT_METRICS_PATH, "a")

    def evaluate():
        model.eval()
        total_loss = total_count = 0
        with torch.no_grad():
            for ids, labs in val_dl:
                ids, labs = ids.to(device), labs.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, labels=labs).loss
                total_loss += loss.item() * ids.size(0)
                total_count += ids.size(0)
        model.train()
        return total_loss / total_count

    step = start_step
    micro = 0
    running_loss = 0.0
    t0 = t_global = time.time()
    best_val_loss = float("inf")

    model.train()
    for epoch in range(n_epochs):
        for ids, labs in train_dl:
            ids, labs = ids.to(device), labs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labs).loss / grad_accum
            loss.backward()
            running_loss += loss.item()
            micro += 1

            if micro % grad_accum != 0:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            current_lr = get_lr(step)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            optimizer.step()
            optimizer.zero_grad()
            step += 1

            if step % log_every == 0:
                dt = time.time() - t0
                avg = running_loss / log_every
                elapsed = time.time() - t_global
                print(f"step {step:>5}/{total_steps} | loss {avg:.4f} | lr {current_lr:.2e} | "
                      f"{elapsed/60:.1f}min")
                mf.write(json.dumps({"step": step, "loss": round(avg, 4),
                                     "lr": current_lr, "elapsed_s": round(elapsed)}) + "\n")
                mf.flush()
                running_loss = 0.0
                t0 = time.time()

            if step % eval_every == 0:
                vl = evaluate()
                ppl = math.exp(min(vl, 20))
                print(f"  [eval] val_loss={vl:.4f} ppl={ppl:.2f}")
                mf.write(json.dumps({"step": step, "val_loss": round(vl, 4),
                                     "ppl": round(ppl, 2)}) + "\n")
                mf.flush()
                if vl < best_val_loss:
                    best_val_loss = vl
                    model.save_pretrained(f"{SFT_CKPT_DIR}/best")
                    print(f"  [best] saved (val_loss={vl:.4f})")
                    volume.commit()

            if step % ckpt_every == 0:
                torch.save({"step": step, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict()},
                           f"{SFT_CKPT_DIR}/sft_ckpt.pt")
                volume.commit()
                print(f"  [ckpt] step {step}")

        print(f"  [epoch {epoch+1}/{n_epochs} done]")

    # Final save
    model.save_pretrained(f"{SFT_CKPT_DIR}/final")
    torch.save({"step": step, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()},
               f"{SFT_CKPT_DIR}/sft_ckpt.pt")

    final_vl = evaluate()
    final_ppl = math.exp(min(final_vl, 20))
    elapsed = time.time() - t_global
    mf.close()
    volume.commit()

    result = {
        "total_steps": step,
        "final_val_loss": round(final_vl, 4),
        "final_ppl": round(final_ppl, 2),
        "best_val_loss": round(best_val_loss, 4),
        "elapsed_min": round(elapsed / 60, 1),
    }
    print(f"SFT complete: {result}")
    return result


@app.local_entrypoint()
def finetune(n_epochs: int = 3):
    """Run SFT fine-tuning."""
    print("Starting SFT fine-tuning...")
    result = run_sft.remote(n_epochs)
    print(f"Done: {result}")
