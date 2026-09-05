#!/usr/bin/env python3
"""Tiny dummy π0.5 run that writes several checkpoints (params-only + full)."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as train_jax
import openpi.training.config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.training import train_log as _train_log


def main() -> None:
    cfg = dataclasses.replace(
        _config.get_config("debug_pi05"),
        batch_size=2,
        num_workers=0,
        num_train_steps=8,
        save_interval=2,
        save_full_interval=4,
        max_to_keep=10,
        keep_period=None,
        log_interval=1,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        checkpoint_base_dir="./checkpoint",
        exp_name="smoke_ckpt_keep",
        ema_decay=0.99,
    )
    train_jax.main(cfg)

    root = Path("checkpoint/smoke_ckpt_keep/checkpoints")
    print("===== STEPS =====")
    for p in sorted(root.iterdir(), key=lambda x: x.name):
        if p.is_symlink() or not p.is_dir() or not p.name.isdigit():
            continue
        ts = (p / "train_state").exists()
        params = (p / "pretrained_model" / "params").exists() or (p / "params").exists()
        print(f"step={p.name} train_state={ts} params={params}")

    import orbax.checkpoint as ocp

    mngr = ocp.CheckpointManager(root)
    print("orbax_steps", sorted(int(s) for s in mngr.all_steps()))
    print("latest_full", _checkpoints.latest_full_step(mngr))
    mngr.close()


if __name__ == "__main__":
    main()
