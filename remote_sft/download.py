"""Download the best SFT checkpoint and retokenized tokenizer from Modal volume."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modal

import config

REMOTE_MODEL = "/checkpoints_sft/best"
REMOTE_TOKENIZER = "/tokenizer"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App(config.PROJECT)
volume = modal.Volume.from_name(config.VOLUME_NAME)


def _download_dir(remote_dir: str) -> int:
    """Download all files from *remote_dir* into LOCAL_DIR. Returns count."""
    entries = volume.listdir(remote_dir)
    count = 0
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        filename = os.path.basename(entry.path)
        local_path = os.path.join(LOCAL_DIR, filename)
        print(f"  {entry.path} -> {filename}")
        with open(local_path, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
        count += 1
    return count


@app.local_entrypoint()
def main():
    """Pull best SFT model + retokenized tokenizer into remote_sft/."""
    print(f"Downloading model from {REMOTE_MODEL} ...")
    n_model = _download_dir(REMOTE_MODEL)

    print(f"Downloading tokenizer from {REMOTE_TOKENIZER} ...")
    n_tok = _download_dir(REMOTE_TOKENIZER)

    print(f"Done — {n_model} model + {n_tok} tokenizer files saved to {LOCAL_DIR}/")
