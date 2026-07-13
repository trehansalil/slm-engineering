"""Download the best SFT checkpoint from the Modal volume to local disk."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import modal

import config

REMOTE_BEST = "/checkpoints_sft/best"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

app = modal.App(config.PROJECT)
volume = modal.Volume.from_name(config.VOLUME_NAME)


@app.local_entrypoint()
def main():
    """Copy the best SFT model files from the Modal volume to remote_model/."""
    entries = volume.listdir(REMOTE_BEST)
    if not entries:
        print(f"No files found at {REMOTE_BEST} on volume '{config.VOLUME_NAME}'")
        raise SystemExit(1)

    print(f"Downloading {len(entries)} files from {REMOTE_BEST} ...")
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        filename = os.path.basename(entry.path)
        remote_path = entry.path
        local_path = os.path.join(LOCAL_DIR, filename)
        print(f"  {entry.path}")
        with open(local_path, "wb") as f:
            for chunk in volume.read_file(remote_path):
                f.write(chunk)

    print(f"Done — best SFT model saved to {LOCAL_DIR}/")
