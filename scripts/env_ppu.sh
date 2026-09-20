#!/usr/bin/env bash
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PPU_SDK_ROOT="${PPU_SDK_ROOT:-/opt/pg1}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/hb/bin/python3}"
export LEROBOT_V21_SRC="${LEROBOT_V21_SRC:-${REPO_DIR}/cigia_lerobot/src}"
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/opt/openpi/models/pi05_fixed/tokenizer.model}"
if [[ ! -f "${PPU_SDK_ROOT}/envsetup.sh" ]]; then
  echo "PPU SDK 未找到: ${PPU_SDK_ROOT}/envsetup.sh" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python 未找到: ${PYTHON_BIN}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ ! -f "${LEROBOT_V21_SRC}/lerobot/datasets/lerobot_dataset.py" ]]; then
  echo "cigia_lerobot 源码未找到: ${LEROBOT_V21_SRC}" >&2
  return 1 2>/dev/null || exit 1
fi
set +u
source "${PPU_SDK_ROOT}/envsetup.sh"
set +u
export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${LEROBOT_V21_SRC}:${PYTHONPATH:-}"
export PALIGEMMA_TOKENIZER_PATH
cd "${REPO_DIR}"
