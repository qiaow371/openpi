#!/usr/bin/env python3
"""Load openpi training overrides from YAML, then start JAX/PyTorch training.

Usage:
  python scripts/train_from_yaml.py --config configs/train_pi05.yaml
  python scripts/train_from_yaml.py --config configs/train_pi05.yaml --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("错误: 需要 PyYAML（pip install pyyaml）") from exc

import openpi.training.config as _config
import openpi.training.schedule as _schedule
import openpi.training.train_log as _train_log


META_KEYS = {"config_name", "framework"}


def _merge_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    """Recursively apply a dict onto a frozen dataclass (unknown keys ignored with warning)."""
    if not updates:
        return obj
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return updates
    field_names = {f.name for f in dataclasses.fields(obj)}
    kwargs: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in field_names:
            print(f"警告: YAML 忽略未知字段: {key}")
            continue
        if value is None:
            continue
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            kwargs[key] = _merge_dataclass(current, value)
        else:
            kwargs[key] = value
    return dataclasses.replace(obj, **kwargs)


def load_train_config(yaml_path: Path) -> tuple[_config.TrainConfig, str]:
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise SystemExit("错误: YAML 顶层必须是映射")

    config_name = raw.get("config_name")
    if not config_name:
        raise SystemExit("错误: YAML 必须提供 config_name（对应 openpi training config.py 中的预设名）")
    framework = str(raw.get("framework", "jax")).lower()
    if framework not in {"jax", "pytorch", "torch"}:
        raise SystemExit("错误: framework 必须是 jax 或 pytorch")
    if framework == "torch":
        framework = "pytorch"

    base = _config.get_config(str(config_name))
    overrides = {k: v for k, v in raw.items() if k not in META_KEYS}
    merged = _merge_dataclass(base, overrides)
    return merged, framework


def main() -> None:
    parser = argparse.ArgumentParser(description="Train openpi from YAML overrides")
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "train_pi05.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="Path to train_pi05.yaml (default: configs/train_pi05.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从已有 checkpoint 续训（覆盖 YAML 里的 resume: false）",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="清空已有实验目录后重训（与 --resume 互斥）",
    )
    args = parser.parse_args()

    if args.resume and args.overwrite:
        raise SystemExit("错误: --resume 与 --overwrite 不能同时使用")

    yaml_path = Path(args.config).resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"错误: 配置文件不存在: {yaml_path}")

    cfg, framework = load_train_config(yaml_path)
    if args.resume:
        cfg = dataclasses.replace(cfg, resume=True, overwrite=False)
    if args.overwrite:
        cfg = dataclasses.replace(cfg, overwrite=True, resume=False)
    # Resolve epoch → step budget before dry-run / training (needs dataset length).
    cfg = _schedule.resolve_epoch_schedule(cfg)
    try:
        cfg = _train_log.resolve_exp_name(cfg)
    except ValueError as exc:
        raise SystemExit(f"错误: {exc}") from exc
    steps_per_epoch = _schedule.steps_per_epoch_from_config(cfg)
    print("==================================================")
    print(f"config_name: {cfg.name}")
    print(f"framework:   {framework}")
    print(f"exp_name:    {cfg.exp_name}")
    print(f"batch_size:  {cfg.batch_size}")
    if cfg.num_epochs is not None:
        print(f"epochs:      {cfg.num_epochs}")
        print(f"steps/ep:    {steps_per_epoch}")
    print(f"steps:       {cfg.num_train_steps}")
    print(f"save_every:  {cfg.save_interval} steps")
    if cfg.save_full_interval is not None:
        print(f"save_full:   {cfg.save_full_interval} steps (train_state)")
    else:
        print("save_full:   every save (params + train_state)")
    print(f"checkpoint:  {cfg.checkpoint_dir}")
    print(f"yaml:        {yaml_path}")
    data_cfg = cfg.data
    prompt_task = bool(getattr(data_cfg, "prompt_from_task", False))
    prompt_sub = bool(getattr(data_cfg, "prompt_from_subtask", False))
    repo_id = getattr(data_cfg, "repo_id", None)
    print(f"repo_id:     {repo_id}")
    print(f"prompt_from_task:    {prompt_task}")
    print(f"prompt_from_subtask: {prompt_sub}")
    print(f"trainable_modules:   {getattr(cfg, 'trainable_modules', 'all')}")
    stage2 = getattr(cfg, "stage2_trainable_modules", None)
    if stage2:
        step = getattr(cfg, "stage2_start_step", None)
        epoch = getattr(cfg, "stage2_start_epoch", None)
        when = f"step {step}" if step is not None else f"epoch {epoch}"
        print(f"stage2_trainable:    {stage2} @ {when}")
    if prompt_sub:
        from openpi.training.data_loader import inspect_subtask_sidecar

        cov = inspect_subtask_sidecar(str(repo_id) if repo_id else None)
        if cov["sidecar"]:
            print(
                f"subtask_sidecar:     {cov['sidecar']}  "
                f"({cov['annotated_episodes']}/{cov['total_episodes']} episodes, {cov['spans']} spans)"
            )
        else:
            print("subtask_sidecar:     MISSING — 仍用 tasks.jsonl 全局 task（先跑 vlm_subtask export）")
    print("==================================================")
    if args.dry_run:
        print("dry-run 完成，未启动训练。")
        return

    # Pass YAML path via env so train.py / train_pytorch can copy it *after*
    # initialize_checkpoint_dir / overwrite wipe (early copy would be deleted).
    os.environ["OPENPI_TRAIN_YAML"] = str(yaml_path)

    if framework == "jax":
        import train as train_jax

        train_jax.main(cfg)
    else:
        import train_pytorch

        train_pytorch.train_loop(cfg)


if __name__ == "__main__":
    # Ensure scripts/ is importable as `train` / `train_pytorch`
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    main()
