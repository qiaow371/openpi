#!/usr/bin/env python3
"""Load openpi training overrides from YAML, then start PyTorch training.

Usage:
  python scripts/train_from_yaml.py --config configs/train_pi05.yaml
  python scripts/train_from_yaml.py --config configs/train_pi05.yaml --dry-run

本分支仅支持 PyTorch（不支持 JAX）。
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
REPO_ROOT = Path(__file__).resolve().parent.parent


def _is_local_data_path(value: str) -> bool:
    """True for filesystem paths; False for HF hub ids / remote URIs."""
    if not value or value == "fake":
        return False
    if value.startswith(("gs://", "s3://", "http://", "https://")):
        return False
    if value.startswith(("/", "~", "./", "../", "data/")):
        return True
    candidate = Path(value)
    if candidate.is_absolute():
        return True
    return (REPO_ROOT / value).exists() or candidate.exists()


def _resolve_local_path(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def resolve_data_paths(cfg: _config.TrainConfig) -> _config.TrainConfig:
    """Resolve relative dataset paths and honor OPENPI_DATASET_DIR / DATASET_DIR."""
    override = os.environ.get("OPENPI_DATASET_DIR") or os.environ.get("DATASET_DIR")
    data = cfg.data
    data_kwargs: dict[str, Any] = {}

    if override:
        # 江算：norm_stats.json 预先写在数据集根目录，assets_dir 与 repo_id 同路径。
        dataset_dir = _resolve_local_path(override)
        data_kwargs["repo_id"] = dataset_dir
        assets = getattr(data, "assets", None)
        if assets is not None:
            data_kwargs["assets"] = dataclasses.replace(
                assets,
                assets_dir=dataset_dir,
                asset_id=getattr(assets, "asset_id", None) or ".",
            )
        else:
            data_kwargs["assets"] = _config.AssetsConfig(assets_dir=dataset_dir, asset_id=".")
    else:
        repo_id = getattr(data, "repo_id", None)
        if isinstance(repo_id, str) and _is_local_data_path(repo_id):
            data_kwargs["repo_id"] = _resolve_local_path(repo_id)
        assets = getattr(data, "assets", None)
        if assets is not None:
            assets_dir = getattr(assets, "assets_dir", None)
            if isinstance(assets_dir, str) and _is_local_data_path(assets_dir):
                data_kwargs["assets"] = dataclasses.replace(
                    assets, assets_dir=_resolve_local_path(assets_dir)
                )

    if not data_kwargs:
        return cfg
    return dataclasses.replace(cfg, data=dataclasses.replace(data, **data_kwargs))


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
    framework = str(raw.get("framework", "pytorch")).lower()
    if framework in {"torch"}:
        framework = "pytorch"
    if framework == "jax":
        raise SystemExit("错误: openpi-jiangsuan 分支仅支持 PyTorch，请将 framework 设为 pytorch")
    if framework != "pytorch":
        raise SystemExit("错误: framework 必须是 pytorch")

    base = _config.get_config(str(config_name))
    overrides = {k: v for k, v in raw.items() if k not in META_KEYS}
    merged = _merge_dataclass(base, overrides)
    merged = resolve_data_paths(merged)
    return merged, framework


def main() -> None:
    parser = argparse.ArgumentParser(description="Train openpi (PyTorch only) from YAML overrides")
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "train_pi05.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_cfg),
        help="Path to train_pi05.yaml (default: configs/train_pi05.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config and exit")
    args = parser.parse_args()

    yaml_path = Path(args.config).resolve()
    if not yaml_path.is_file():
        raise SystemExit(f"错误: 配置文件不存在: {yaml_path}")

    cfg, framework = load_train_config(yaml_path)
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
    print(f"checkpoint:  {cfg.checkpoint_dir}")
    print(f"assets_dirs: {cfg.assets_dirs}")
    print(f"repo_id:     {getattr(cfg.data, 'repo_id', None)}")
    print(f"yaml:        {yaml_path}")
    print("==================================================")
    if args.dry_run:
        print("dry-run 完成，未启动训练。")
        return

    os.environ["OPENPI_TRAIN_YAML"] = str(yaml_path)

    import train_pytorch

    train_pytorch.train_loop(cfg)


if __name__ == "__main__":
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    main()
