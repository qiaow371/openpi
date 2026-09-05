"""Helpers to persist OpenPI training logs / hyperparameters for multi-run comparison."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


TRAIN_LOG_NAME = "train.log"
TRAIN_CONFIG_NAME = "train_config.json"
TRAIN_HARDWARE_NAME = "train_hardware.json"
RUN_META_NAME = "run_meta.json"
CONFIG_JSON_NAME = "config.json"
CPU_GPU_USAGE_NAME = "cpu_gpu_usage.json"
TRAIN_LOSS_HISTORY_NAME = "train_loss_history.json"
TRAIN_LOSS_PLOT_NAME = "train_loss.png"
SOURCE_YAML_ENV = "OPENPI_TRAIN_YAML"

# Checkpoint layout (aligned with ACT / LeRobot-style):
#   {run_dir}/checkpoints/{padded_step}/pretrained_model/
#   {run_dir}/checkpoints/last -> {padded_step}
#   {run_dir}/resource_monitor/
# Auto exp_name:
#   {policy}_chunk_{N}_lr_{lr}_epochs_{E}[_{abbr}_{val}...]_{YYYYMMDD}
CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
LAST_CHECKPOINT_LINK = "last"
RESOURCE_MONITOR_DIR = "resource_monitor"


def asset_checkpoint_relpath(asset_id: str) -> str:
    """Map ``asset_id`` to a relative path that stays under an assets directory.

    ``asset_id="."`` means use the assets directory itself (no subfolder).
    """
    text = str(asset_id).strip()
    if text in {"", "."}:
        return ""
    path = Path(text)
    if path.is_absolute():
        name = path.name
        if not name or name in {".", ".."}:
            raise ValueError(f"Invalid absolute asset_id: {asset_id}")
        return name
    parts = [part for part in path.parts if part not in ("", ".")]
    if not parts:
        return ""
    if any(part == ".." for part in parts):
        raise ValueError(f"Invalid asset_id (path escape): {asset_id}")
    return str(Path(*parts))


def _format_lr_for_dirname(lr: Any) -> str:
    """Format learning rate for folder names, e.g. 2.5e-05 -> 2.5e-5."""
    try:
        value = float(lr)
    except (TypeError, ValueError):
        return str(lr).replace("/", "_")
    text = f"{value:.1e}"
    if "e-" in text:
        base, exp = text.split("e-", 1)
        text = f"{base}e-{int(exp)}"
    elif "e+" in text:
        base, exp = text.split("e+", 1)
        text = f"{base}e+{int(exp)}"
    return text


def _policy_run_tag(config: Any) -> str:
    """Short policy name used in auto exp_name folders (aligned with ACT naming)."""
    name = str(getattr(config, "name", "") or "").lower()
    model = getattr(config, "model", None)
    if getattr(model, "pi05", False) or "pi05" in name:
        return "pi05"
    if "pi0_fast" in name or "fast" in name:
        return "pi0_fast"
    if name.startswith("pi0") or "pi0" in name:
        return "pi0"
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
    return cleaned or "openpi"


def _cfg_get(config: Any, *path: str, default: Any = None) -> Any:
    cur: Any = config
    for part in path:
        if cur is None:
            return default
        cur = getattr(cur, part, None)
    return default if cur is None else cur


# Baseline matches configs/train_pi05.yaml on this branch (num_workers: 0 for PPU).
# Strings / bools are ignored. Core keys are skipped in the diff section.
_ExpField = tuple[str, str, Callable[[Any], Any], int | float]

_EXP_NAME_NUMERIC_FIELDS: list[_ExpField] = [
    ("batch_size", "bs", lambda c: _cfg_get(c, "batch_size"), 16),
    ("num_workers", "nw", lambda c: _cfg_get(c, "num_workers"), 0),
    ("seed", "seed", lambda c: _cfg_get(c, "seed"), 42),
    ("log_interval", "li", lambda c: _cfg_get(c, "log_interval"), 100),
    ("save_every_epochs", "see", lambda c: _cfg_get(c, "save_every_epochs"), 10),
    ("save_interval", "si", lambda c: _cfg_get(c, "save_interval"), 1000),
    ("keep_period", "kp", lambda c: _cfg_get(c, "keep_period"), 5000),
    ("fsdp_devices", "fsdp", lambda c: _cfg_get(c, "fsdp_devices"), 1),
    ("ema_decay", "ema", lambda c: _cfg_get(c, "ema_decay"), 0.99),
    ("action_horizon", "chunk", lambda c: _cfg_get(c, "model", "action_horizon"), 50),
    ("peak_lr", "lr", lambda c: _cfg_get(c, "lr_schedule", "peak_lr"), 2.5e-5),
    ("warmup_steps", "wu", lambda c: _cfg_get(c, "lr_schedule", "warmup_steps"), 1000),
    ("decay_lr", "dlr", lambda c: _cfg_get(c, "lr_schedule", "decay_lr"), 2.5e-6),
    ("weight_decay", "wd", lambda c: _cfg_get(c, "optimizer", "weight_decay"), 1.0e-10),
    ("b1", "b1", lambda c: _cfg_get(c, "optimizer", "b1"), 0.9),
    ("b2", "b2", lambda c: _cfg_get(c, "optimizer", "b2"), 0.95),
    ("eps", "eps", lambda c: _cfg_get(c, "optimizer", "eps"), 1.0e-8),
    ("clip_gradient_norm", "cgn", lambda c: _cfg_get(c, "optimizer", "clip_gradient_norm"), 1.0),
    ("num_epochs", "epochs", lambda c: _cfg_get(c, "num_epochs"), 100),
    ("num_train_steps", "steps", lambda c: _cfg_get(c, "num_train_steps"), 30_000),
]

_EXP_NAME_CORE_KEYS = frozenset({"action_horizon", "peak_lr", "num_epochs", "num_train_steps"})
_EXP_NAME_SCI_KEYS = frozenset({"peak_lr", "decay_lr", "weight_decay", "eps"})


def _is_numeric_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            return False
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return False


def _as_python_number(value: Any) -> int | float:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except Exception:
            pass
    if isinstance(value, bool):
        raise TypeError("bool is not a numeric hyperparam")
    if isinstance(value, int):
        return int(value)
    return float(value)


def _numeric_values_equal(actual: Any, default: Any) -> bool:
    if not _is_numeric_scalar(actual) or not _is_numeric_scalar(default):
        return False
    a = _as_python_number(actual)
    d = _as_python_number(default)
    if isinstance(a, float) or isinstance(d, float):
        return math.isclose(float(a), float(d), rel_tol=0.0, abs_tol=1e-12)
    return int(a) == int(d)


def _format_numeric_for_dirname(key: str, value: Any) -> str:
    if not _is_numeric_scalar(value):
        return str(value).replace("/", "_").replace(" ", "")
    value = _as_python_number(value)
    if key in _EXP_NAME_SCI_KEYS or (
        isinstance(value, float) and (abs(value) < 1e-3 or abs(value) >= 1e4)
    ):
        return _format_lr_for_dirname(value)
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text.replace(".", "p") if "." in text else text
    return str(int(value))


def collect_numeric_exp_name_diffs(config: Any) -> list[tuple[str, str]]:
    """Return ``[(abbrev, formatted_value), ...]`` for numeric fields ≠ YAML baseline."""
    num_epochs = _cfg_get(config, "num_epochs")
    use_epochs = num_epochs is not None and _is_numeric_scalar(num_epochs) and int(num_epochs) > 0
    save_every = _cfg_get(config, "save_every_epochs")
    use_save_every = save_every is not None and _is_numeric_scalar(save_every)

    diffs: list[tuple[str, str]] = []
    for key, abbr, getter, default in _EXP_NAME_NUMERIC_FIELDS:
        if key in _EXP_NAME_CORE_KEYS:
            continue
        # After resolve_epoch_schedule, num_train_steps / save_interval are derived — skip.
        if key == "num_train_steps" and use_epochs:
            continue
        if key == "save_interval" and use_save_every:
            continue
        if key == "save_every_epochs" and not use_save_every:
            continue
        actual = getter(config)
        if not _is_numeric_scalar(actual):
            continue
        if _numeric_values_equal(actual, default):
            continue
        diffs.append((abbr, _format_numeric_for_dirname(key, actual)))
    return diffs


def _cap_dirname(parts: list[str], *, max_bytes: int = 240) -> str:
    name = "_".join(parts)
    encoded = name.encode("utf-8")
    if len(encoded) <= max_bytes:
        return name
    digest = hashlib.sha1(encoded).hexdigest()[:8]
    core = parts[:4]
    date = parts[-1]
    kept: list[str] = []
    for seg in parts[4:-1]:
        candidate = "_".join(core + kept + [seg] + [f"x{digest}", date])
        if len(candidate.encode("utf-8")) > max_bytes:
            break
        kept.append(seg)
    return "_".join(core + kept + [f"x{digest}", date])


def build_exp_name(config: Any, *, now: datetime | None = None) -> str:
    """Build ``{policy}_chunk_{N}_lr_{lr}_epochs_{E}[_{abbr}_{val}...]_{YYYYMMDD}``.

    Fixed prefix always includes chunk / lr / epochs|steps. Extra segments are
    numeric hyperparameters that differ from this branch's train_pi05.yaml
    baseline (strings and bools ignored). Names are capped near Linux NAME_MAX.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d")
    model = getattr(config, "model", None)
    chunk = int(getattr(model, "action_horizon", 0) or 0)
    lr = _format_lr_for_dirname(_cfg_get(config, "lr_schedule", "peak_lr", default=0.0))
    tag = _policy_run_tag(config)
    num_epochs = getattr(config, "num_epochs", None)
    if num_epochs is not None and int(num_epochs) > 0:
        parts = [tag, f"chunk_{chunk}", f"lr_{lr}", f"epochs_{int(num_epochs)}"]
    else:
        steps = int(getattr(config, "num_train_steps", 0) or 0)
        parts = [tag, f"chunk_{chunk}", f"lr_{lr}", f"steps_{steps}"]
    for abbr, value in collect_numeric_exp_name_diffs(config):
        parts.append(f"{abbr}_{value}")
    parts.append(stamp)
    return _cap_dirname(parts)


def _exp_name_is_missing(exp_name: Any) -> bool:
    if exp_name is None:
        return True
    text = str(exp_name).strip()
    if not text or text.upper() == "MISSING":
        return True
    if "MISSING" in type(exp_name).__name__:
        return True
    return text.lower() in {"auto", "null", "none"}


def resolve_exp_name(config: Any) -> Any:
    """Auto-fill ``exp_name`` when YAML leaves it unset/auto.

    Fresh run::
        {policy}_chunk_{N}_lr_{lr}_epochs_{E}[_{abbr}_{val}...]_{YYYYMMDD}
        collision on same day → append ``_HHMMSS``

    Resume::
        ``exp_name`` must already be the concrete run name.
    """
    if not dataclasses.is_dataclass(config):
        raise TypeError(f"Expected TrainConfig dataclass, got {type(config)}")

    if bool(getattr(config, "resume", False)):
        if _exp_name_is_missing(getattr(config, "exp_name", None)):
            raise ValueError(
                "resume=true 时必须在 YAML 中设置具体的 exp_name（指向已有 run 目录名），不能留空或 auto"
            )
        return config

    if not _exp_name_is_missing(getattr(config, "exp_name", None)):
        return config

    name = build_exp_name(config)
    updated = dataclasses.replace(config, exp_name=name)
    checkpoint_dir = Path(updated.checkpoint_dir)
    if checkpoint_dir.exists() and not bool(getattr(config, "overwrite", False)):
        name = f"{name}_{datetime.now().strftime('%H%M%S')}"
        updated = dataclasses.replace(config, exp_name=name)
    logging.info("Auto exp_name: %s", updated.exp_name)
    return updated


def to_jsonable(obj: Any) -> Any:
    """Recursively convert configs / pathlib / nested structures into JSON-serializable values."""
    try:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, dict):
            return {str(k): to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_jsonable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__fspath__"):
            return str(obj)
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")
        item = getattr(obj, "item", None)
        if callable(item):
            try:
                return to_jsonable(item())
            except Exception:
                pass
        return repr(obj)
    except Exception as exc:  # never let config dump crash training
        return f"<unserializable {type(obj).__name__}: {exc}>"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
        f.write("\n")
    return path


def resolve_source_yaml(source_yaml: str | Path | None = None) -> Path | None:
    if source_yaml:
        path = Path(source_yaml)
        return path if path.is_file() else None
    env = os.environ.get(SOURCE_YAML_ENV, "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    return None


def copy_source_yaml(output_dir: Path | str, source_yaml: str | Path | None = None) -> Path | None:
    """Copy user YAML into the run dir after checkpoint wipe; keep basename + train.yaml alias."""
    src = resolve_source_yaml(source_yaml)
    if src is None:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dst = output_dir / src.name
    shutil.copy2(src, dst)
    alias = output_dir / "train.yaml"
    if dst.resolve() != alias.resolve():
        shutil.copy2(src, alias)
    return dst


def save_run_meta(
    output_dir: Path | str,
    *,
    source_yaml: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)
    meta: dict[str, Any] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "argv": list(sys.argv),
        "config_file": str(src.resolve()) if src else None,
        "config_basename": src.name if src else None,
    }
    if extra:
        meta.update(extra)
    return write_json(output_dir / RUN_META_NAME, meta)


def save_train_config(
    output_dir: Path | str,
    config: Any,
    *,
    source_yaml: str | Path | None = None,
) -> Path:
    """Write resolved training hyperparameters to ``train_config.json`` under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)
    payload = {
        "_meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config_file": str(src.resolve()) if src else None,
            "config_basename": src.name if src else None,
            "note": "Resolved effective TrainConfig after YAML overrides on the named preset.",
        },
        "params": to_jsonable(config),
    }
    return write_json(output_dir / TRAIN_CONFIG_NAME, payload)


def build_policy_config(config: Any) -> dict[str, Any]:
    """Compact policy-facing ``config.json`` for a checkpoint bundle."""
    model = getattr(config, "model", None)
    return {
        "type": type(model).__name__ if model is not None else None,
        "name": getattr(config, "name", None),
        "exp_name": getattr(config, "exp_name", None),
        "policy_metadata": to_jsonable(getattr(config, "policy_metadata", None)),
        "model": to_jsonable(model),
        "batch_size": getattr(config, "batch_size", None),
        "num_train_steps": getattr(config, "num_train_steps", None),
        "save_interval": getattr(config, "save_interval", None),
    }


def save_policy_config(output_dir: Path | str, config: Any) -> Path:
    return write_json(Path(output_dir) / CONFIG_JSON_NAME, build_policy_config(config))


def _read_cpu_mem() -> dict[str, Any]:
    load1 = load5 = load15 = None
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            parts = f.read().split()
            load1, load5, load15 = float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass

    mem: dict[str, Any] = {}
    try:
        info: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                num = raw.strip().split()[0]
                if num.isdigit():
                    info[key] = int(num)
        total = info.get("MemTotal")
        avail = info.get("MemAvailable")
        if total and avail is not None:
            used = total - avail
            mem = {
                "total_kb": total,
                "used_kb": used,
                "available_kb": avail,
                "used_pct": round(100.0 * used / total, 2),
            }
    except Exception:
        pass

    cpu_count = os.cpu_count()
    return {
        "cpu_count": cpu_count,
        "loadavg": {"1m": load1, "5m": load5, "15m": load15},
        "memory": mem,
    }


def _read_gpu_usage() -> list[dict[str, Any]]:
    query = (
        "index,name,utilization.gpu,utilization.memory,"
        "memory.used,memory.total,temperature.gpu,power.draw"
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return [{"error": f"nvidia-smi unavailable: {exc}"}]

    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        return [{"error": err or f"nvidia-smi exit {result.returncode}"}]

    gpus: list[dict[str, Any]] = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        gpus.append(
            {
                "index": int(parts[0]) if parts[0].isdigit() else parts[0],
                "name": parts[1],
                "utilization_gpu_pct": _maybe_float(parts[2]),
                "utilization_memory_pct": _maybe_float(parts[3]),
                "memory_used_mb": _maybe_float(parts[4]),
                "memory_total_mb": _maybe_float(parts[5]),
                "temperature_c": _maybe_float(parts[6]),
                "power_draw_w": _maybe_float(parts[7]),
            }
        )
    return gpus


def _maybe_float(value: str) -> float | str | None:
    try:
        return float(value)
    except Exception:
        return value if value else None


def collect_cpu_gpu_usage() -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu": _read_cpu_mem(),
        "gpu": _read_gpu_usage(),
    }


def save_cpu_gpu_usage(output_dir: str | Path) -> Path:
    payload = collect_cpu_gpu_usage()
    return write_json(Path(output_dir) / CPU_GPU_USAGE_NAME, payload)


def save_train_hardware(output_dir: Path | str, extra: dict[str, Any] | None = None) -> Path:
    """Write a small hardware / launcher manifest next to train.log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "cpu_gpu_usage": collect_cpu_gpu_usage(),
    }
    if extra:
        payload.update(extra)
    return write_json(output_dir / TRAIN_HARDWARE_NAME, payload)


class ResourceMonitor:
    """Background sampler writing cpu/gpu usage under ``{output_dir}/resource_monitor``."""

    def __init__(self, output_dir: str | Path, *, interval_s: float = 5.0):
        self.output_dir = Path(output_dir)
        self.monitor_dir = self.output_dir / RESOURCE_MONITOR_DIR
        self.interval_s = max(1.0, float(interval_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> ResourceMonitor:
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "interval_s": self.interval_s,
            "hostname": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        write_json(self.monitor_dir / "meta.json", meta)
        save_cpu_gpu_usage(self.output_dir)

        gpu_csv = self.monitor_dir / "gpu_util.csv"
        if not gpu_csv.exists():
            gpu_csv.write_text(
                "timestamp,gpu_index,name,util_gpu_pct,util_mem_pct,mem_used_mb,mem_total_mb,temp_c,power_w\n",
                encoding="utf-8",
            )
        cpu_log = self.monitor_dir / "cpu_mem.log"
        if not cpu_log.exists():
            cpu_log.write_text("", encoding="utf-8")

        self._thread = threading.Thread(target=self._loop, name="openpi-resource-monitor", daemon=True)
        self._thread.start()
        logging.info("Resource monitor started: %s", self.monitor_dir)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 2)
            self._thread = None
        usage = collect_cpu_gpu_usage()
        write_json(self.output_dir / CPU_GPU_USAGE_NAME, usage)
        hw_path = self.output_dir / TRAIN_HARDWARE_NAME
        try:
            payload: dict[str, Any] = {}
            if hw_path.is_file():
                with hw_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        payload = loaded
            payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
            payload["cpu_gpu_usage"] = usage
            write_json(hw_path, payload)
        except Exception:
            logging.exception("Failed to refresh %s on monitor stop", TRAIN_HARDWARE_NAME)
        logging.info("Resource monitor stopped: %s", self.monitor_dir)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:
                logging.exception("Resource monitor sample failed")
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        snap = collect_cpu_gpu_usage()
        ts = snap["timestamp"]
        cpu = snap.get("cpu") or {}
        mem = cpu.get("memory") or {}
        load = cpu.get("loadavg") or {}
        line = (
            f"{ts} load1={load.get('1m')} load5={load.get('5m')} load15={load.get('15m')} "
            f"mem_used_pct={mem.get('used_pct')} mem_used_kb={mem.get('used_kb')} "
            f"mem_total_kb={mem.get('total_kb')}\n"
        )
        with (self.monitor_dir / "cpu_mem.log").open("a", encoding="utf-8") as f:
            f.write(line)

        rows = []
        for gpu in snap.get("gpu") or []:
            if "error" in gpu:
                continue
            rows.append(
                f"{ts},{gpu.get('index')},{gpu.get('name')},"
                f"{gpu.get('utilization_gpu_pct')},{gpu.get('utilization_memory_pct')},"
                f"{gpu.get('memory_used_mb')},{gpu.get('memory_total_mb')},"
                f"{gpu.get('temperature_c')},{gpu.get('power_draw_w')}\n"
            )
        if rows:
            with (self.monitor_dir / "gpu_util.csv").open("a", encoding="utf-8") as f:
                f.writelines(rows)

        write_json(self.output_dir / CPU_GPU_USAGE_NAME, snap)


def start_resource_monitor(output_dir: str | Path, *, interval_s: float = 5.0) -> ResourceMonitor:
    return ResourceMonitor(output_dir, interval_s=interval_s).start()


def get_step_identifier(step: int, total_steps: int) -> str:
    """Zero-pad step id; width follows total_steps (min 6)."""
    num_digits = max(6, len(str(int(total_steps))))
    return f"{int(step):0{num_digits}d}"


def get_checkpoints_dir(run_dir: str | Path) -> Path:
    return Path(run_dir) / CHECKPOINTS_DIR


def get_step_checkpoint_dir(run_dir: str | Path, total_steps: int, step: int) -> Path:
    return get_checkpoints_dir(run_dir) / get_step_identifier(step, total_steps)


def get_pretrained_model_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / PRETRAINED_MODEL_DIR


def get_training_state_dir(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / TRAINING_STATE_DIR


def update_checkpoint_link(checkpoint_dir: Path, link_name: str) -> Path:
    """Point ``checkpoints/{link_name}`` at the given step directory (relative symlink)."""
    checkpoint_dir = Path(checkpoint_dir)
    link_path = checkpoint_dir.parent / link_name
    if link_path.is_symlink() or link_path.is_file():
        link_path.unlink()
    elif link_path.exists():
        logging.warning("Skip updating %s: path exists and is not a symlink (%s)", link_name, link_path)
        return link_path
    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    link_path.symlink_to(relative_target)
    return link_path


def _has_model_weights(directory: Path) -> bool:
    return (directory / "model.safetensors").is_file() or (directory / "params").exists()


def resolve_pretrained_model_dir(checkpoint_dir: str | Path) -> Path:
    """
    Resolve a directory that contains inference weights (``params/`` or ``model.safetensors``).

    Accepts any of:
      - ``.../pretrained_model``
      - ``.../checkpoints/{step}`` or ``.../checkpoints/last``
      - run root (prefers ``checkpoints/last/pretrained_model``)
      - legacy flat step dir with ``params/`` or ``model.safetensors`` at top level
    """
    root = Path(checkpoint_dir)
    candidates = [
        root,
        root / PRETRAINED_MODEL_DIR,
        root / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR,
        root / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK,
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        nested = candidate / PRETRAINED_MODEL_DIR
        if nested.is_dir() and _has_model_weights(nested):
            return nested.resolve() if nested.is_symlink() else nested
        if _has_model_weights(candidate):
            return candidate.resolve() if candidate.is_symlink() else candidate

    search_roots = []
    ckpts = root / CHECKPOINTS_DIR
    if ckpts.is_dir():
        search_roots.append(ckpts)
    search_roots.append(root)

    for search in search_roots:
        step_dirs: list[tuple[int, Path]] = []
        for child in search.iterdir():
            if child.name == LAST_CHECKPOINT_LINK:
                continue
            if child.name.isdigit() and child.exists():
                step_dirs.append((int(child.name), child))
        for _, step_dir in sorted(step_dirs, key=lambda x: x[0], reverse=True):
            nested = step_dir / PRETRAINED_MODEL_DIR
            if nested.is_dir() and _has_model_weights(nested):
                return nested.resolve() if nested.is_symlink() else nested
            if _has_model_weights(step_dir):
                return step_dir.resolve() if step_dir.is_symlink() else step_dir

    return root


def list_checkpoint_steps(run_or_checkpoints_dir: str | Path) -> list[int]:
    """List numeric checkpoint steps under a run dir or its ``checkpoints/`` folder."""
    root = Path(run_or_checkpoints_dir)
    search_roots: list[Path] = []
    ckpts = root / CHECKPOINTS_DIR
    if ckpts.is_dir():
        search_roots.append(ckpts)
    search_roots.append(root)

    steps: list[int] = []
    for search in search_roots:
        for child in search.iterdir():
            if not child.name.isdigit():
                continue
            if child.is_dir() or child.is_symlink():
                steps.append(int(child.name))
        if steps:
            break
    return sorted(set(steps))


def resolve_step_dir(run_dir: str | Path, step: int, total_steps: int | None = None) -> Path:
    """Locate ``checkpoints/{step}`` (or legacy ``{run}/{step}``) supporting padded names."""
    run_dir = Path(run_dir)
    search_roots = [get_checkpoints_dir(run_dir), run_dir]
    names: list[str] = []
    if total_steps is not None:
        names.append(get_step_identifier(step, total_steps))
    names.append(str(int(step)))
    for width in (6, 7, 8, len(str(int(step)))):
        padded = f"{int(step):0{width}d}"
        if padded not in names:
            names.append(padded)

    for search in search_roots:
        if not search.is_dir():
            continue
        for name in names:
            path = search / name
            if path.exists():
                return path.resolve() if path.is_symlink() else path
    raise FileNotFoundError(f"No checkpoint directory for step {step} under {run_dir}")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        return float(value)
    except Exception:
        return None


def setup_train_file_logging(output_dir: Path | str, *, append: bool = True) -> Path:
    """Attach a FileHandler that writes to ``{output_dir}/train.log`` (console handlers kept)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / TRAIN_LOG_NAME

    logger = logging.getLogger()
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    resolved = str(log_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == resolved:
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a" if append else "w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    if logger.handlers and logger.handlers[0].formatter is not None:
        file_handler.setFormatter(logger.handlers[0].formatter)
    else:
        file_handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(file_handler)
    return log_path


def format_metrics_line(
    step: int,
    *,
    loss: float | None = None,
    grad_norm: float | None = None,
    lr: float | None = None,
    update_s: float | None = None,
    data_s: float | None = None,
    epoch: float | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build a LeRobot-compatible metrics line: ``step:N loss:... grdn:... lr:...``."""
    parts = [f"step:{int(step)}"]
    epoch_f = _as_float(epoch)
    loss_f = _as_float(loss)
    grad_f = _as_float(grad_norm)
    lr_f = _as_float(lr)
    updt_f = _as_float(update_s)
    data_f = _as_float(data_s)
    if epoch_f is not None:
        parts.append(f"epch:{epoch_f:.4g}")
    if loss_f is not None:
        parts.append(f"loss:{loss_f:.6g}")
    if grad_f is not None:
        parts.append(f"grdn:{grad_f:.6g}")
    if lr_f is not None:
        parts.append(f"lr:{lr_f:.6g}")
    if updt_f is not None:
        parts.append(f"updt_s:{updt_f:.6g}")
    if data_f is not None:
        parts.append(f"data_s:{data_f:.6g}")
    if extra:
        for key, value in extra.items():
            value_f = _as_float(value)
            if value_f is not None:
                parts.append(f"{key}:{value_f:.6g}")
    return " ".join(parts)


def load_train_loss_history(output_dir: Path | str) -> tuple[list[int], list[float]]:
    """Load persisted train-loss history for resume / continued plotting."""
    path = Path(output_dir) / TRAIN_LOSS_HISTORY_NAME
    if not path.is_file():
        return [], []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        steps = [int(s) for s in payload.get("steps", [])]
        losses = [float(v) for v in payload.get("losses", [])]
        if len(steps) != len(losses):
            n = min(len(steps), len(losses))
            return steps[:n], losses[:n]
        return steps, losses
    except Exception:
        logging.exception("Failed to load %s", path)
        return [], []


def save_train_loss_plot(
    output_dir: Path | str,
    steps: list[int] | list[float],
    losses: list[float],
    *,
    filename: str = TRAIN_LOSS_PLOT_NAME,
    title: str = "train_loss",
) -> Path | None:
    """
    Save a train-loss curve with finer y ticks and a marked minimum.

    Also writes ``train_loss_history.json`` next to the plot so resume can continue.
    Never raises into the training loop.
    """
    output_dir = Path(output_dir)
    if not steps or not losses or len(steps) != len(losses):
        return None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        history_path = output_dir / TRAIN_LOSS_HISTORY_NAME
        with history_path.open("w", encoding="utf-8") as f:
            json.dump({"steps": list(steps), "losses": list(losses)}, f)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.ticker import FormatStrFormatter, MaxNLocator

        xs = np.asarray(steps, dtype=float)
        ys = np.asarray(losses, dtype=float)
        plot_path = output_dir / filename
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label="train", color="C0")

        train_min = float(np.min(ys))
        train_min_step = float(xs[int(np.argmin(ys))])
        ax.axhline(train_min, color="green", linestyle="--", linewidth=1.2, alpha=0.85)
        ax.scatter([train_min_step], [train_min], color="green", s=36, zorder=5)
        ax.text(
            1.01,
            train_min,
            f"{train_min:.5f}",
            transform=ax.get_yaxis_transform(),
            color="green",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
            clip_on=False,
        )

        ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.subplots_adjust(right=0.86)
        fig.savefig(plot_path)
        plt.close(fig)
        return plot_path
    except Exception:
        logging.exception("Failed to save train loss plot")
        return None


def init_train_log(
    output_dir: Path | str,
    config: Any,
    *,
    append: bool = True,
    hardware_extra: dict[str, Any] | None = None,
    source_yaml: str | Path | None = None,
) -> Path:
    """
    Prepare run artifacts (call **after** checkpoint-dir overwrite wipe):

    - ``train.log``
    - ``train_hardware.json`` / ``cpu_gpu_usage.json``
    - ``run_meta.json``
    - copied source YAML (basename + ``train.yaml`` alias)

    Note: ``config.json`` / ``train_config.json`` are only written inside each
    ``checkpoints/*/pretrained_model/`` bundle, not at the run root.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    src = resolve_source_yaml(source_yaml)

    log_path = setup_train_file_logging(output_dir, append=append)
    yaml_dst = None
    try:
        yaml_dst = copy_source_yaml(output_dir, src)
    except Exception:
        logging.exception("Failed to copy source yaml into output dir")

    hw_extra = dict(hardware_extra or {})
    if src is not None:
        hw_extra.setdefault("config_file", str(src.resolve()))
        hw_extra.setdefault("config_basename", src.name)
    try:
        hw_path = save_train_hardware(output_dir, hw_extra)
        usage_path = save_cpu_gpu_usage(output_dir)
    except Exception:
        logging.exception("Failed to save hardware / cpu-gpu usage")
        hw_path = output_dir / TRAIN_HARDWARE_NAME
        usage_path = output_dir / CPU_GPU_USAGE_NAME

    try:
        meta_path = save_run_meta(
            output_dir,
            source_yaml=src,
            extra={
                "framework": hw_extra.get("framework"),
                "exp_name": getattr(config, "exp_name", None),
                "config_name": getattr(config, "name", None),
                "checkpoint_dir": str(getattr(config, "checkpoint_dir", output_dir)),
            },
        )
    except Exception:
        logging.exception("Failed to save %s", RUN_META_NAME)
        meta_path = output_dir / RUN_META_NAME

    logging.info("Output dir: %s", output_dir)
    logging.info("Train log: %s", log_path)
    logging.info("Train hardware: %s", hw_path)
    logging.info("CPU/GPU usage: %s", usage_path)
    logging.info("Run meta: %s", meta_path)
    if yaml_dst is not None:
        logging.info("Source yaml copied: %s", yaml_dst)
    logging.info("cfg.steps=%s", getattr(config, "num_train_steps", None))
    try:
        logging.info("%s", json.dumps(to_jsonable(config), indent=2, ensure_ascii=False))
    except Exception:
        logging.exception("Failed to dump train config into train.log")
    return log_path
