#!/usr/bin/env bash
# Install only the incremental Python deps needed for OpenPI PI0.5 on PPU.
# Safe for shared Jiangsuan images: skips packages already present, never
# upgrades torch / torchvision / torchaudio, and refuses to change torch version.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/lerobot/.venv/bin/python}"
PPU_SDK_ROOT="${PPU_SDK_ROOT:-/usr/local/PPU_SDK}"
# Public mirror for incremental deps (h5py/jax/...).
# Do NOT read ambient PIP_INDEX_URL: PPU images / envsetup.sh export an
# authenticated art-pub.eng.t-head.cn index for +ppu torch; inheriting it
# makes pip prompt for username. Override only via EXTRA_PIP_INDEX_URL.
EXTRA_PIP_INDEX_URL="${EXTRA_PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
EXTRA_PIP_TRUSTED_HOST="${EXTRA_PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
DRY_RUN="${DRY_RUN:-0}"

# import_name|pip_spec
# Versions match the working PPU container (Jul 2026).
# Do NOT list torch / torchvision / numpy / transformers / lerobot / openpi here.
EXTRA_DEPS=(
  "h5py|h5py==3.16.0"
  "jax|jax==0.5.3"
  "jaxlib|jaxlib==0.5.3"
  "flax|flax==0.10.2"
  "optax|optax==0.2.4"
  "orbax.checkpoint|orbax-checkpoint==0.11.13"
  "chex|chex==0.1.89"
  "etils|etils==1.14.0"
  "tensorstore|tensorstore==0.1.84"
  "toolz|toolz==1.1.0"
  "augmax|augmax==0.4.1"
  "jaxtyping|jaxtyping==0.2.36"
  "beartype|beartype==0.19.0"
  "opt_einsum|opt_einsum==3.4.0"
)

log() { printf '[setup_ppu_extra_deps] %s\n' "$*"; }
die() { printf '[setup_ppu_extra_deps] ERROR: %s\n' "$*" >&2; exit 1; }

torch_version() {
  # Prefer distribution metadata (keeps +ppu suffix); fall back to __version__.
  "${PYTHON_BIN}" -c '
import importlib.metadata as m
try:
    print(m.version("torch"))
except Exception:
    import torch
    print(torch.__version__)
' 2>/dev/null | tail -n 1
}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  die "Python not found or not executable: ${PYTHON_BIN}"
fi

if [[ -f "${PPU_SDK_ROOT}/envsetup.sh" ]]; then
  # Keep base PPU libs on PATH/LD_LIBRARY_PATH so torch import stays quiet/correct.
  set +u
  # shellcheck disable=SC1090
  source "${PPU_SDK_ROOT}/envsetup.sh"
  set -u
  log "sourced ${PPU_SDK_ROOT}/envsetup.sh"
else
  log "WARNING: ${PPU_SDK_ROOT}/envsetup.sh not found; continue without it"
fi

log "python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -V

TORCH_BEFORE="$(torch_version)" || die "torch import failed; refuse to modify a broken base env"
[[ -n "${TORCH_BEFORE}" ]] || die "empty torch version; refuse to modify env"
log "torch (before): ${TORCH_BEFORE}"

if [[ "${TORCH_BEFORE}" != *"+ppu"* ]]; then
  log "WARNING: torch is not a +ppu build (${TORCH_BEFORE}); continue anyway"
fi

need_install=()
for item in "${EXTRA_DEPS[@]}"; do
  import_name="${item%%|*}"
  pip_spec="${item#*|}"
  if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${import_name}")
PY
  then
    ver="$("${PYTHON_BIN}" - <<PY
import importlib
m = importlib.import_module("${import_name}")
print(getattr(m, "__version__", "ok"))
PY
)"
    log "skip  ${import_name} (${ver})"
  else
    log "need  ${pip_spec}"
    need_install+=("${pip_spec}")
  fi
done

if [[ ${#need_install[@]} -eq 0 ]]; then
  log "all incremental deps already present; nothing to do"
  exit 0
fi

log "will install ${#need_install[@]} package(s): ${need_install[*]}"
if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1, exit without installing"
  exit 0
fi

# only-if-needed: do not upgrade already-satisfied packages (esp. torch).
# Explicitly avoid pulling torch as a dependency upgrade.
# Strip ambient PIP_* index env vars so they cannot override --index-url.
log "pip index: ${EXTRA_PIP_INDEX_URL}"
PIP_ARGS=(
  install
  --upgrade-strategy only-if-needed
  --index-url "${EXTRA_PIP_INDEX_URL}"
  --trusted-host "${EXTRA_PIP_TRUSTED_HOST}"
  --disable-pip-version-check
)

env -u PIP_INDEX_URL -u PIP_EXTRA_INDEX_URL \
  "${PYTHON_BIN}" -m pip "${PIP_ARGS[@]}" "${need_install[@]}"

TORCH_AFTER="$(torch_version)"
log "torch (after):  ${TORCH_AFTER}"
if [[ "${TORCH_BEFORE}" != "${TORCH_AFTER}" ]]; then
  die "torch version changed (${TORCH_BEFORE} -> ${TORCH_AFTER}); base env may be damaged"
fi

log "verify imports"
fail=0
for item in "${EXTRA_DEPS[@]}"; do
  import_name="${item%%|*}"
  if ! "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${import_name}")
PY
  then
    log "FAIL import ${import_name}"
    fail=1
  fi
done
[[ "${fail}" -eq 0 ]] || die "some packages failed to import after install"

log "done"
