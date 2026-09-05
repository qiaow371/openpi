"""Compute OpenPI normalization stats directly from a LeRobot v2.1 dataset.

This intentionally reads only parquet state/action columns. Images and language
are irrelevant to normalization, so avoiding video decoding makes this suitable
for large AV1 datasets.

Does **not** load the PaliGemma tokenizer (not needed for state/action stats).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.policies import aloha_policy
from openpi.shared import normalize
from openpi.training import config as config_lib
import openpi.transforms as _transforms


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="pi05_aloha")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="LeRobot v2.1 数据集根目录（与 YAML 的 data.repo_id 一致；不是 cigia_lerobot）",
    )
    return parser.parse_args()


def _stats(values: np.ndarray) -> normalize.NormStats:
    values = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    return normalize.NormStats(
        mean=values.mean(axis=0),
        std=values.std(axis=0),
        q01=np.quantile(values, 0.01, axis=0),
        q99=np.quantile(values, 0.99, axis=0),
    )


def _data_transforms_for_norm(data_factory: config_lib.DataConfigFactory) -> tuple:
    """Mirror ``LeRobotAlohaDataConfig.create`` data transforms without loading the tokenizer."""
    if not isinstance(data_factory, config_lib.LeRobotAlohaDataConfig):
        raise TypeError(
            f"compute_norm_stats_v21 currently supports LeRobotAlohaDataConfig, got {type(data_factory).__name__}"
        )
    transforms_list: list = [aloha_policy.AlohaInputs(adapt_to_pi=data_factory.adapt_to_pi)]
    if data_factory.use_delta_joint_actions:
        delta_action_mask = _transforms.make_bool_mask(*[int(x) for x in data_factory.delta_action_dims])
        transforms_list.append(_transforms.DeltaActions(delta_action_mask))
    return tuple(transforms_list)


def main() -> None:
    args = _parse_args()
    dataset_dir = args.dataset_dir.resolve()
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {info.get('codebase_version')!r}")

    # cfg = config_lib.get_config(args.config)
    config_path = Path(args.config).expanduser()

    if config_path.is_file():
        from train_from_yaml import load_train_config

        cfg, _ = load_train_config(config_path.resolve())
    else:
        cfg = config_lib.get_config(args.config)
        
    # Avoid cfg.data.create(...): that builds ModelTransformFactory → PaligemmaTokenizer,
    # which requires PALIGEMMA_TOKENIZER_PATH even though norm stats only need data transforms.
    transforms = _data_transforms_for_norm(cfg.data)
    horizon = cfg.model.action_horizon

    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    dummy_images = {
        "base_0_rgb": np.zeros((3, 1, 1), dtype=np.uint8),
        "left_wrist_0_rgb": np.zeros((3, 1, 1), dtype=np.uint8),
        "right_wrist_0_rgb": np.zeros((3, 1, 1), dtype=np.uint8),
    }
    parquet_files = sorted((dataset_dir / "data").glob("chunk-*/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files under {dataset_dir / 'data'}")

    for episode_number, parquet_path in enumerate(parquet_files, start=1):
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
        if states.shape != actions.shape or states.ndim != 2:
            raise ValueError(
                f"Unexpected state/action shapes in {parquet_path}: {states.shape}, {actions.shape}"
            )

        last = len(actions) - 1
        offsets = np.arange(horizon)
        for frame_index, state in enumerate(states):
            action_indices = np.minimum(frame_index + offsets, last)
            item = {
                "state": state.copy(),
                "actions": actions[action_indices].copy(),
                "images": dummy_images,
            }
            for transform in transforms:
                item = transform(item)
            all_states.append(np.asarray(item["state"], dtype=np.float64))
            all_actions.append(np.asarray(item["actions"], dtype=np.float64))

        print(f"[{episode_number}/{len(parquet_files)}] {parquet_path.name}: {len(states)} frames")

    state_values = np.stack(all_states)
    action_values = np.concatenate(all_actions, axis=0)
    norm_stats = {"state": _stats(state_values), "actions": _stats(action_values)}
    normalize.save(dataset_dir, norm_stats)
    print(f"Wrote {dataset_dir / 'norm_stats.json'}")
    print(f"state samples={len(state_values)}, shape={state_values.shape}")
    print(f"action samples={len(action_values)}, shape={action_values.shape}")
    print(f"state q01={norm_stats['state'].q01.tolist()}")
    print(f"state q99={norm_stats['state'].q99.tolist()}")
    print(f"actions q01={norm_stats['actions'].q01.tolist()}")
    print(f"actions q99={norm_stats['actions'].q99.tolist()}")


if __name__ == "__main__":
    main()
