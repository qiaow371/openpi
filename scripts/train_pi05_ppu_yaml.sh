#!/usr/bin/env bash
# 江算 π₀.₅：自动加载 cigia_lerobot 源码环境，再按 YAML 训练。
# 数据集路径读 configs/train_pi05.yaml 里的 data.repo_id（与 cigia_lerobot 无关）。
# 用法:
#   bash scripts/train_pi05_ppu_yaml.sh
#   bash scripts/train_pi05_ppu_yaml.sh --dry-run
#   bash scripts/train_pi05_ppu_yaml.sh --config configs/train_pi05.yaml
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_DIR}/scripts/env_ppu.sh"

CONFIG="${REPO_DIR}/configs/train_pi05.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

exec "${PYTHON_BIN}" "${REPO_DIR}/scripts/train_from_yaml.py" --config "${CONFIG}" "${EXTRA_ARGS[@]}"
