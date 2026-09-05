"""Offline π0.5 val on frozen val episodes (jiangsuan-16).

Same dataloader as train (prompt_from_subtask). Loss is FM MSE, not CE.
Do not run while the 16-GPU train job is occupying all cards.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-batches", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))

    import safetensors.torch

    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.training import data_loader as _data
    from openpi.training import train_log as _train_log
    from scripts.train_from_yaml import load_train_config

    cfg, _fw = load_train_config(Path(args.config))
    splits = json.loads(Path(args.splits).read_text())
    allow = {int(x) for x in (splits.get("val_episodes") or [])}
    if not allow:
        raise SystemExit(f"{args.splits} 没有 val_episodes")

    ckpt_root = Path(args.ckpt_dir)
    steps = _train_log.list_checkpoint_steps(ckpt_root)
    if not steps:
        raise SystemExit(f"没有 step checkpoint: {ckpt_root}")
    latest = max(steps)
    step_dir = _train_log.resolve_step_dir(ckpt_root, latest, cfg.num_train_steps)
    weights = _train_log.get_pretrained_model_dir(step_dir) / "model.safetensors"
    if not weights.is_file():
        raise SystemExit(f"missing {weights}")

    cfg = dataclasses_replace_batch(cfg, args.batch_size)
    loader = _data.create_data_loader(
        cfg,
        framework="pytorch",
        shuffle=False,
        num_batches=args.max_batches,
        episode_allowlist=allow,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = PI0Pytorch(cfg.model).to(device)
    safetensors.torch.load_model(model, str(weights), strict=False)
    model.eval()

    losses: list[float] = []
    with torch.no_grad():
        for bi, (observation, actions) in enumerate(loader):
            observation = _tree_to(observation, device)
            actions = actions.to(device=device, dtype=torch.float32)
            loss = model(observation, actions)
            val = float(loss.mean().item())
            losses.append(val)
            logging.info("val batch %s loss=%.5f", bi, val)

    mean = sum(losses) / len(losses) if losses else None
    summary = {
        "kind": "pi05_offline_val",
        "host": "jiangsuan-16",
        "ckpt": str(weights),
        "step": latest,
        "val_episodes": len(allow),
        "batches": len(losses),
        "mean_fm_loss": mean,
        "losses": losses,
        "prompt_from_subtask": True,
        "note": "FM MSE on val episodes; not CE; real-robot eval stays on infer-pc.",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    rec = Path("/nvme1n1/vlm-sft/outputs/annotate/pipeline_record.json")
    hist = Path("/nvme1n1/vlm-sft/outputs/annotate/pipeline_record_history.jsonl")
    if rec.is_file():
        obj = json.loads(rec.read_text())
        obj["val"] = summary
        rec.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
        with hist.open("a") as f:
            f.write(json.dumps({"val": summary}, ensure_ascii=False) + "\n")
    logging.info("wrote %s mean_fm_loss=%s", out / "val_summary.json", mean)


def dataclasses_replace_batch(cfg, batch_size: int):
    import dataclasses

    return dataclasses.replace(cfg, batch_size=batch_size, num_workers=0)


def _tree_to(obs, device):
    if hasattr(obs, "to"):
        return obs.to(device)
    import jax

    return jax.tree.map(lambda x: x.to(device) if hasattr(x, "to") else x, obs)


if __name__ == "__main__":
    main()
