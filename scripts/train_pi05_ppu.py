"""PPU launcher for the existing PyTorch PI0.5 trainer."""

import argparse
import dataclasses

from openpi.training import config as config_lib
from scripts import train_pytorch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PI0.5 with the PPU PyTorch runtime")
    parser.add_argument("config", help="OpenPI config name, for example pi05_aloha or debug_pi05")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--checkpoint-base-dir")
    parser.add_argument("--pytorch-weight-path")
    parser.add_argument("--precision", choices=("bfloat16", "float32"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = config_lib.get_config(args.config)
    overrides = {
        "exp_name": args.exp_name,
        "num_workers": args.num_workers,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb,
    }
    optional = {
        "batch_size": args.batch_size,
        "num_train_steps": args.num_train_steps,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "checkpoint_base_dir": args.checkpoint_base_dir,
        "pytorch_weight_path": args.pytorch_weight_path,
        "pytorch_training_precision": args.precision,
    }
    overrides.update({key: value for key, value in optional.items() if value is not None})
    train_pytorch.init_logging()
    train_pytorch.train_loop(dataclasses.replace(config, **overrides))


if __name__ == "__main__":
    main()
