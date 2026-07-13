"""SFT fine-tuning for the 125M SLM on Modal.

Tokenizes chat JSONL, masks loss on system/user turns, fine-tunes from
the pre-trained base checkpoint with LoRA-free full fine-tuning.

Logs train_loss, val_loss, and perplexity per epoch. Checkpoints every
2 epochs by default.

Usage:
    modal run sft/finetune_modal.py::tokenize_sft
    modal run sft/finetune_modal.py::finetune
    modal run sft/finetune_modal.py::finetune --n-epochs 20
    modal run sft/finetune_modal.py::finetune --n-epochs 20 --fresh
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
    timeout=3600 * 4,
)
def run_sft(
    n_epochs: int = 20,
    ckpt_every_epochs: int = 2,
    fresh: bool = False,
) -> dict:
    """Fine-tune the pre-trained base model on the SFT dataset.

    Logs train_loss, val_loss, perplexity per epoch. Checkpoints model,
    optimizer, and training state every ``ckpt_every_epochs`` epochs.
    """
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, random_split
    from transformers import LlamaForCausalLM

    device = torch.device("cuda")

    # ── data ──────────────────────────────────────────────────────────
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

    # ── hyperparams ───────────────────────────────────────────────────
    lr = 2e-5
    min_lr = 2e-6
    batch_size = 16
    grad_accum = 2
    weight_decay = 0.01
    grad_clip = 1.0
    log_every = 20

    # ── model: resume or fresh from base ──────────────────────────────
    from huggingface_hub import snapshot_download

    if not os.path.exists(f"{BASE_CKPT_DIR}/config.json"):
        print(f"Downloading base model from {HF_BASE_REPO}...")
        snapshot_download(
            repo_id=HF_BASE_REPO,
            local_dir=BASE_CKPT_DIR,
            ignore_patterns=["*.md", ".gitattributes"],
        )
        volume.commit()
        print("Base model downloaded and cached on volume")

    start_epoch = 0
    resume_ckpt = None

    if not fresh:
        ckpt_dirs = []
        if os.path.exists(SFT_CKPT_DIR):
            ckpt_dirs = sorted(
                [d for d in os.listdir(SFT_CKPT_DIR) if d.startswith("epoch_")],
                key=lambda x: int(x.split("_")[1]),
            )
        if ckpt_dirs:
            latest = ckpt_dirs[-1]
            resume_path = f"{SFT_CKPT_DIR}/{latest}"
            state_path = f"{resume_path}/training_state.pt"
            if os.path.exists(state_path):
                resume_ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
                start_epoch = resume_ckpt["epoch"]
                print(f"Resuming from {resume_path} (after epoch {start_epoch}, "
                      f"val_loss={resume_ckpt.get('val_loss', '?')}, "
                      f"ppl={resume_ckpt.get('ppl', '?')})")

    if resume_ckpt is not None:
        model = LlamaForCausalLM.from_pretrained(
            f"{SFT_CKPT_DIR}/epoch_{start_epoch:02d}", torch_dtype=torch.bfloat16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(BASE_CKPT_DIR, torch_dtype=torch.bfloat16)
        print(f"Loaded fresh base model from {HF_BASE_REPO}")

    model = model.to(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params ({n_params/1e6:.1f}M)")

    # ── dataloaders ───────────────────────────────────────────────────
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False,
                        num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay,
    )
    if resume_ckpt is not None and "optimizer" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        print("Restored optimizer state")

    remaining_epochs = n_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Already trained {start_epoch} epochs (target {n_epochs}). Nothing to do.")
        return {"status": "already_done", "epochs_completed": start_epoch}

    steps_per_epoch = len(train_dl) // grad_accum
    total_steps = steps_per_epoch * remaining_epochs
    warmup_steps = min(100, total_steps // 10)

    def get_lr(step):
        if step < warmup_steps:
            return lr * step / max(1, warmup_steps)
        ratio = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * ratio))

    print(f"\nSFT config: epochs {start_epoch + 1} -> {n_epochs}, "
          f"{steps_per_epoch} steps/epoch, {total_steps} total steps")
    print(f"batch={batch_size}x{grad_accum}, lr={lr}, warmup={warmup_steps}")
    print(f"Checkpoint every {ckpt_every_epochs} epochs\n")

    # ── metrics files ─────────────────────────────────────────────────
    os.makedirs(SFT_CKPT_DIR, exist_ok=True)
    epoch_log_path = f"{SFT_CKPT_DIR}/epoch_metrics.jsonl"
    step_log_path = SFT_METRICS_PATH

    step_log = open(step_log_path, "a")
    epoch_log = open(epoch_log_path, "a")

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

    def save_checkpoint(epoch, step, val_loss, ppl):
        ckpt_path = f"{SFT_CKPT_DIR}/epoch_{epoch:02d}"
        os.makedirs(ckpt_path, exist_ok=True)
        model.save_pretrained(ckpt_path)
        torch.save({
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "val_loss": val_loss,
            "ppl": ppl,
        }, f"{ckpt_path}/training_state.pt")
        volume.commit()
        print(f"  [ckpt] saved epoch {epoch} -> {ckpt_path}")

    # ── initial eval ──────────────────────────────────────────────────
    init_vl = evaluate()
    init_ppl = math.exp(min(init_vl, 20))
    print(f"Initial val_loss={init_vl:.4f}  ppl={init_ppl:.2f}")
    epoch_log.write(json.dumps({
        "epoch": start_epoch, "train_loss": None,
        "val_loss": round(init_vl, 4), "ppl": round(init_ppl, 2),
        "phase": "start",
    }) + "\n")
    epoch_log.flush()

    # ── training loop ─────────────────────────────────────────────────
    step = 0
    micro = 0
    best_val_loss = init_vl
    avg_train_loss = vl = init_vl
    ppl = init_ppl
    t0 = t_global = time.time()
    running_loss_step = 0.0

    model.train()
    for epoch_idx in range(remaining_epochs):
        current_epoch = start_epoch + epoch_idx + 1
        epoch_train_loss = 0.0
        epoch_train_steps = 0

        for ids, labs in train_dl:
            ids, labs = ids.to(device), labs.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, labels=labs).loss / grad_accum
            loss.backward()
            running_loss_step += loss.item()
            epoch_train_loss += loss.item()
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
            epoch_train_steps += 1

            if step % log_every == 0:
                avg = running_loss_step / log_every
                elapsed = time.time() - t_global
                print(f"  step {step:>5}/{total_steps} | loss {avg:.4f} | "
                      f"lr {current_lr:.2e} | {elapsed/60:.1f}min")
                step_log.write(json.dumps({
                    "epoch": current_epoch, "step": step,
                    "loss": round(avg, 4), "lr": current_lr,
                    "elapsed_s": round(elapsed),
                }) + "\n")
                step_log.flush()
                running_loss_step = 0.0
                t0 = time.time()

        # ── end-of-epoch eval ─────────────────────────────────────────
        avg_train_loss = epoch_train_loss / max(1, epoch_train_steps)
        vl = evaluate()
        ppl = math.exp(min(vl, 20))
        elapsed = time.time() - t_global
        improving = "+" if vl < best_val_loss else "-"

        print(f"[epoch {current_epoch:>2}/{n_epochs}] "
              f"train_loss={avg_train_loss:.4f}  val_loss={vl:.4f}  "
              f"ppl={ppl:.2f} {improving}  | {elapsed/60:.1f}min")

        epoch_log.write(json.dumps({
            "epoch": current_epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(vl, 4),
            "ppl": round(ppl, 2),
            "improving": vl < best_val_loss,
            "elapsed_min": round(elapsed / 60, 1),
        }) + "\n")
        epoch_log.flush()

        if vl < best_val_loss:
            best_val_loss = vl
            best_path = f"{SFT_CKPT_DIR}/best"
            os.makedirs(best_path, exist_ok=True)
            model.save_pretrained(best_path)
            volume.commit()
            print(f"  [best] saved (val_loss={vl:.4f})")

        if current_epoch % ckpt_every_epochs == 0:
            save_checkpoint(current_epoch, step, vl, ppl)

    # ── final save ────────────────────────────────────────────────────
    final_path = f"{SFT_CKPT_DIR}/final"
    os.makedirs(final_path, exist_ok=True)
    model.save_pretrained(final_path)
    save_checkpoint(n_epochs, step, vl, ppl)

    step_log.close()
    epoch_log.close()
    volume.commit()

    elapsed = time.time() - t_global
    best_ppl = math.exp(min(best_val_loss, 20))
    result = {
        "epochs": n_epochs,
        "total_steps": step,
        "final_train_loss": round(avg_train_loss, 4),
        "final_val_loss": round(vl, 4),
        "final_ppl": round(ppl, 2),
        "best_val_loss": round(best_val_loss, 4),
        "best_ppl": round(best_ppl, 2),
        "elapsed_min": round(elapsed / 60, 1),
        "epoch_log": epoch_log_path,
    }
    print(f"\nSFT complete: {json.dumps(result, indent=2)}")
    return result


@app.local_entrypoint()
def finetune(n_epochs: int = 20, ckpt_every_epochs: int = 2, fresh: bool = False):
    """Run SFT fine-tuning.

    Args:
        n_epochs: Total epochs to train (default 20).
        ckpt_every_epochs: Save checkpoint every N epochs (default 2).
        fresh: Ignore existing SFT checkpoints and start from base model.
    """
    print(f"Starting SFT: {n_epochs} epochs, ckpt every {ckpt_every_epochs}, "
          f"fresh={fresh}")
    result = run_sft.remote(n_epochs, ckpt_every_epochs, fresh)
    import json as json_mod
    print(f"\nDone: {json_mod.dumps(result, indent=2)}")
