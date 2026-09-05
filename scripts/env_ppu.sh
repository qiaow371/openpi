#!/usr/bin/env bash
# 江算 / PPU 公共环境：默认绑定本仓库 cigia_lerobot（代码，不是数据）。
# 用法: source scripts/env_ppu.sh
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PPU_SDK_ROOT="${PPU_SDK_ROOT:-/usr/local/PPU_SDK}"
export PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/lerobot/.venv/bin/python}"
# cigia_lerobot = LeRobot v2.1 源码；训练数据仍由 YAML 的 data.repo_id 指定
export LEROBOT_V21_SRC="${LEROBOT_V21_SRC:-${REPO_DIR}/cigia_lerobot/src}"
export PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-/home/ubuntu/lerobot/CQTongBot-train-ppu/base_model/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c/tokenizer.model}"

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
  echo "请确认已 git pull，且仓库内存在 cigia_lerobot/src" >&2
  return 1 2>/dev/null || exit 1
fi

# envsetup 可能引用未定义变量
set +u
# shellcheck disable=SC1090
source "${PPU_SDK_ROOT}/envsetup.sh"
set +u

export PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/packages/openpi-client/src:${LEROBOT_V21_SRC}:${PYTHONPATH:-}"
export PALIGEMMA_TOKENIZER_PATH
cd "${REPO_DIR}"
