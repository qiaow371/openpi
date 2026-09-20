#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PPU_SDK_ROOT="${PPU_SDK_ROOT:-/usr/local/PPU_SDK}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/lerobot/.venv/bin/python}"
PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/home/ubuntu/lerobot/CQTongBot-train-ppu/base_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer.model}"
# 默认使用本仓库自带的 cigia_lerobot（LeRobot v2.1）
LEROBOT_V21_SRC="${LEROBOT_V21_SRC:-${REPO_DIR}/cigia_lerobot/src}"

if [[ ! -f "${PPU_SDK_ROOT}/envsetup.sh" ]]; then
    echo "PPU SDK setup script not found: ${PPU_SDK_ROOT}/envsetup.sh" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Existing PI0.5 Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${LEROBOT_V21_SRC}/lerobot/datasets/lerobot_dataset.py" ]]; then
    echo "Existing LeRobot v2.1 source not found: ${LEROBOT_V21_SRC}" >&2
    echo "Expected bundled path: ${REPO_DIR}/cigia_lerobot/src" >&2
    exit 1
fi

set +u
source "${PPU_SDK_ROOT}/envsetup.sh"
set -u

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${LEROBOT_V21_SRC}:${PYTHONPATH:-}"
export PALIGEMMA_TOKENIZER_PATH
cd "${REPO_DIR}"
exec "${PYTHON_BIN}" scripts/train_pi05_ppu.py "$@"
