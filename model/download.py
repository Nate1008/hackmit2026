# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from huggingface_hub import snapshot_download


def setup_checkpoints():
    Path("checkpoints").mkdir(exist_ok=True)
    try:
        snapshot_download(
            repo_id="facebook/ShapeR",
            allow_patterns=["*.ckpt", "*.yaml", "*.pth"],
            local_dir="./checkpoints",
        )
    except Exception as e:
        print(f"Error downloading checkpoints: {e}")
        print(
            "Are you sure you are logged in to huggingface? (`huggingface_cli login`)"
        )
        exit(1)
