"""OpenPI π0.5 VLM 独立入口。

  python -m vlm_subtask compose --config vlm_subtask/configs/00086.yaml --data-dir ...
  python -m vlm_subtask train   --config vlm_subtask/configs/00086.yaml --data-dir ... --model ...
  python -m vlm_subtask val     --config vlm_subtask/configs/00086.yaml --data-dir ... --model ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from vlm_subtask.compose import compose_rows, read_jsonl, split_jsonl_dir, write_splits
from vlm_subtask.export_lerobot import export_from_compose


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def cmd_compose(args: argparse.Namespace) -> None:
    cfg = _load_yaml(Path(args.config))
    templates = cfg.get("subtask_templates")
    if not templates:
        raise SystemExit("config 缺少 subtask_templates")
    data_dir = Path(args.data_dir)
    segs = read_jsonl(data_dir / "segments.jsonl")
    if not segs:
        raise SystemExit(f"没有 {data_dir}/segments.jsonl，先跑 annotate_pipeline prepare")
    verify_rows = read_jsonl(data_dir / "verify.jsonl")
    rows, summary = compose_rows(
        segs,
        task=cfg["task"],
        templates=templates,
        verify_rows=verify_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    write_splits(data_dir, rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_export(args: argparse.Namespace) -> None:
    summary = export_from_compose(Path(args.data_dir), Path(args.repo))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def cmd_train(args: argparse.Namespace) -> None:
    from vlm_subtask.train_val import run_train

    cfg = _load_yaml(Path(args.config))
    templates = cfg.get("subtask_templates") or {}
    data_dir = Path(args.data_dir)
    split_dir = split_jsonl_dir(data_dir)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else data_dir / "vlm_ckpt"
    print(f"[vlm] splits {split_dir}", flush=True)
    run_train(
        data_dir=data_dir,
        split_dir=split_dir,
        model_path=args.model,
        templates=templates,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        log_every=args.log_every,
        ckpt_dir=ckpt_dir,
    )


def cmd_val(args: argparse.Namespace) -> None:
    from vlm_subtask.train_val import run_val

    cfg = _load_yaml(Path(args.config))
    data_dir = Path(args.data_dir)
    split_dir = split_jsonl_dir(data_dir)
    out = Path(args.out) if args.out else data_dir / "vlm_val_summary.json"
    run_val(
        data_dir=data_dir,
        split_dir=split_dir,
        model_path=args.model,
        templates=cfg.get("subtask_templates") or {},
        device=args.device,
        out_path=out,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="π0.5 VLM 短 subtask：compose / train / val（独立于 JAX 动作训练）")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compose", help="segments.jsonl → vlm_train/val.jsonl")
    c.add_argument("--config", required=True)
    c.add_argument("--data-dir", required=True, help="含 segments.jsonl 与 sheets/ 的目录")
    c.add_argument("--val-ratio", type=float, default=0.15)
    c.add_argument("--seed", type=int, default=42)
    c.set_defaults(func=cmd_compose)

    e = sub.add_parser("export", help="写成 AgiBot action_config sidecar（task_info + episodes_detailed_task.jsonl）")
    e.add_argument("--data-dir", required=True, help="compose 产出目录（含 cot.jsonl）")
    e.add_argument("--repo", required=True, help="LeRobot 数据集根目录")
    e.set_defaults(func=cmd_export)

    t = sub.add_parser("train", help="冻 vision，训语言头，每个 epoch val")
    t.add_argument("--config", required=True)
    t.add_argument("--data-dir", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--ckpt-dir", default=None)
    t.add_argument("--device", default="cuda:0")
    t.add_argument("--epochs", type=int, default=3)
    t.add_argument("--batch-size", type=int, default=2)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--log-every", type=int, default=10)
    t.set_defaults(func=cmd_train)

    v = sub.add_parser("val", help="生成 subtask，写 exact_match / label_acc")
    v.add_argument("--config", required=True)
    v.add_argument("--data-dir", required=True)
    v.add_argument("--model", required=True)
    v.add_argument("--out", default=None)
    v.add_argument("--device", default="cuda:0")
    v.set_defaults(func=cmd_val)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
