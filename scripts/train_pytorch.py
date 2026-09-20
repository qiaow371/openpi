"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data
import openpi.training.schedule as _schedule
import openpi.training.train_log as _train_log


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        step_dir = _train_log.get_step_checkpoint_dir(config.checkpoint_dir, config.num_train_steps, global_step)
        pretrained_dir = _train_log.get_pretrained_model_dir(step_dir)
        checkpoints_root = _train_log.get_checkpoints_dir(config.checkpoint_dir)
        checkpoints_root.mkdir(parents=True, exist_ok=True)
        tmp_ckpt_dir = checkpoints_root / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_pretrained = tmp_ckpt_dir / _train_log.PRETRAINED_MODEL_DIR
        tmp_training_state = tmp_ckpt_dir / _train_log.TRAINING_STATE_DIR
        tmp_pretrained.mkdir(parents=True, exist_ok=True)
        tmp_training_state.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_pretrained / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_training_state / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_training_state / "metadata.pt")

        norm_stats = data_config.norm_stats
        if norm_stats is not None:
            _normalize.save(tmp_ckpt_dir / "assets", norm_stats)

        try:
            _train_log.save_policy_config(tmp_pretrained, config)
            _train_log.save_train_config(tmp_pretrained, config)
        except Exception:
            logging.exception("Failed to write config files into checkpoint bundle")

        # Atomically move temp directory to final location
        if step_dir.exists():
            shutil.rmtree(step_dir)
        tmp_ckpt_dir.rename(step_dir)
        _train_log.update_checkpoint_link(step_dir, _train_log.LAST_CHECKPOINT_LINK)

        logging.info(f"Saved checkpoint at step {global_step} -> {pretrained_dir}")
        logging.info(f"Checkpoint policy after step {global_step}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device, *, total_steps: int | None = None):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = _train_log.list_checkpoint_steps(checkpoint_dir)
    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    step_dir = _train_log.resolve_step_dir(checkpoint_dir, latest_step, total_steps)
    pretrained_dir = _train_log.get_pretrained_model_dir(step_dir)
    training_state_dir = _train_log.get_training_state_dir(step_dir)
    # Fall back to flat step dir (legacy OpenPI layout).
    if not (pretrained_dir / "model.safetensors").exists() and (step_dir / "model.safetensors").exists():
        pretrained_dir = step_dir
        training_state_dir = step_dir
    ckpt_dir = step_dir

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = pretrained_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = training_state_dir / "optimizer.pt"
        if not optimizer_path.exists():
            optimizer_path = pretrained_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata_path = training_state_dir / "metadata.pt"
        if not metadata_path.exists():
            metadata_path = pretrained_dir / "metadata.pt"
        metadata = torch.load(metadata_path, map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    steps = _train_log.list_checkpoint_steps(checkpoint_dir)
    return max(steps) if steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


# ---------------------------------------------------------------------------
# per-joint × per-timestep loss 探针（2026-09-14 华为 π0.5 事故复盘）
# ---------------------------------------------------------------------------
# 标量 loss = mean([B, H, D]) 会掩盖所有结构性问题。这里保留 [H, D] 矩阵落盘。
# 判据：D=32 中后 16 维是 pad 0，其目标 u = ε，而 x_t = t·ε 使 ε 完全可知，
# 所以预训练好的 π0.5 pad 维 loss 已接近 0 —— pad_mean ≈ 1 等价于"动作专家是冷的"。
_PROBE_REAL_DIM = 16
_PROBE_PAD_MEAN_COLD = 0.5


def _loss_probe_stats(mat) -> dict:
    """[H, D] numpy 矩阵 → 可直接喂 LLM 的摘要 dict。"""
    real = mat[:, :_PROBE_REAL_DIM]
    pad = mat[:, _PROBE_REAL_DIM:]
    dim = real.mean(axis=0)
    return {
        "dim_real": [round(float(v), 5) for v in dim],
        "per_step": [round(float(v), 5) for v in mat.mean(axis=1)],
        "matrix": [[round(float(v), 5) for v in row] for row in mat],
        "real_mean": round(float(real.mean()), 5),
        "pad_mean": round(float(pad.mean()), 5),
        "worst_dim": int(dim.argmax()),
        "best_dim": int(dim.argmin()),
        "verdict": "cold-expert" if float(pad.mean()) > _PROBE_PAD_MEAN_COLD else "warm",
    }


def _write_loss_probe(checkpoint_dir: str, step, mat) -> None:
    """把 [H, D] 均值矩阵追加成一行 JSON。任何异常都只 warning —— 探针不许弄死训练。"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "loss_probe.jsonl")
        rec = {"step": int(step)}
        rec.update(_loss_probe_stats(mat))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        logging.info(
            "PROBE step=%s real=%.4f pad=%.4f worst=d%d(%.3f) best=d%d(%.3f) verdict=%s",
            step,
            rec["real_mean"],
            rec["pad_mean"],
            rec["worst_dim"],
            rec["dim_real"][rec["worst_dim"]],
            rec["best_dim"],
            rec["dim_real"][rec["best_dim"]],
            rec["verdict"],
        )
    except Exception as e:  # noqa: BLE001 - 探针失败不能影响训练
        logging.warning("loss probe 写入失败（忽略）：%s", e)


class ActionDimAbort(RuntimeError):
    """逐维 loss 门禁触发。训练循环应让它冒泡退出，不要吞掉。"""


DIM_NAMES = [f"L{j}" for j in range(1, 8)] + [f"R{j}" for j in range(1, 8)] + ["Lgrip", "Rgrip"]
REAL_DIM = 16
GUARD_UNTIL_STEP = 100
# 健康 run 起训单维 <1.4、max/median ≤3.1；有毒 00036 起训单维 12~15、比值 35~40。
ABS_ABORT = 5.0
RATIO_ABORT = 8.0
RATIO_MIN_MEDIAN = 0.05
PAD_ABORT = 0.1
SCALAR_ABORT = 1.0


def _evaluate_action_dims(mat, step=0, scalar_loss=None) -> dict:
    """[H, D] 矩阵 → {abort, codes, reasons, ...}。不抛错。"""
    arr = np.asarray(mat, dtype=np.float64)
    step = int(step)
    codes: list[str] = []
    reasons: list[str] = []

    if arr.ndim != 2 or arr.size == 0:
        codes.append("bad_shape")
        reasons.append(f"探针矩阵形状异常：{getattr(arr, 'shape', None)}")
        return {
            "abort": True,
            "step": step,
            "codes": codes,
            "reasons": reasons,
            "dim_real": [],
            "pad_mean": None,
            "real_mean": None,
            "scalar_loss": None if scalar_loss is None else float(scalar_loss),
            "worst_dim": None,
            "worst_name": None,
            "worst_val": None,
            "median": None,
            "ratio": None,
            "window": step <= GUARD_UNTIL_STEP,
        }

    if not np.isfinite(arr).all():
        codes.append("nan")
        reasons.append("逐维 loss 出现 NaN/Inf")

    n_real = min(REAL_DIM, arr.shape[1])
    real = arr[:, :n_real]
    dim = real.mean(axis=0)
    pad = arr[:, n_real:] if arr.shape[1] > n_real else None
    pad_mean = float(pad.mean()) if pad is not None and pad.size else None
    real_mean = float(real.mean())
    worst = int(dim.argmax()) if dim.size else 0
    worst_val = float(dim[worst]) if dim.size else None
    median = float(np.median(dim)) if dim.size else None
    ratio = (
        float(worst_val / median)
        if worst_val is not None and median is not None and median >= RATIO_MIN_MEDIAN
        else None
    )
    name = DIM_NAMES[worst] if worst < len(DIM_NAMES) else f"d{worst}"
    scalar = None if scalar_loss is None else float(scalar_loss)

    if worst_val is not None and worst_val >= ABS_ABORT:
        codes.append("dim_abs")
        reasons.append(
            f"{name}(d{worst})={worst_val:.3f} ≥ {ABS_ABORT}，单维爆炸"
            "（stats/掩码拆开的典型形状，标量会被 pad 摊掉）"
        )

    in_window = step <= GUARD_UNTIL_STEP
    if in_window and ratio is not None and ratio >= RATIO_ABORT:
        codes.append("dim_ratio")
        reasons.append(
            f"前 {GUARD_UNTIL_STEP} 步 {name}/median = {ratio:.1f} ≥ {RATIO_ABORT}"
            f"（median={median:.3f}）"
        )

    if pad_mean is not None and pad_mean >= PAD_ABORT:
        codes.append("pad_cold")
        reasons.append(
            f"pad_mean={pad_mean:.4f} ≥ {PAD_ABORT}，动作专家冷启动（热基座应接近 0）"
        )

    if scalar is not None and scalar >= SCALAR_ABORT:
        codes.append("scalar_cold")
        reasons.append(f"标量 loss={scalar:.4f} ≥ {SCALAR_ABORT}，坏基座哨兵")

    dim_real = [round(float(v), 5) for v in dim]
    return {
        "abort": bool(codes),
        "step": step,
        "codes": codes,
        "reasons": reasons,
        "dim_real": dim_real,
        "dim_names": DIM_NAMES[:n_real],
        "pad_mean": None if pad_mean is None else round(pad_mean, 5),
        "real_mean": round(real_mean, 5),
        "scalar_loss": None if scalar is None else round(scalar, 5),
        "worst_dim": worst,
        "worst_name": name,
        "worst_val": None if worst_val is None else round(worst_val, 5),
        "median": None if median is None else round(median, 5),
        "ratio": None if ratio is None else round(ratio, 3),
        "window": in_window,
        "thresholds": {
            "abs": ABS_ABORT,
            "ratio": RATIO_ABORT,
            "until_step": GUARD_UNTIL_STEP,
            "pad": PAD_ABORT,
            "scalar": SCALAR_ABORT,
        },
    }


def _format_action_dim_abort_md(result: dict) -> str:
    names = result.get("dim_names") or DIM_NAMES
    dims = result.get("dim_real") or []
    lines = [
        "# π0.5 action-dim 门禁：训练已停止",
        "",
        f"- step: {result.get('step')}",
        f"- codes: {', '.join(result.get('codes') or []) or '（无）'}",
        f"- worst: {result.get('worst_name')}(d{result.get('worst_dim')})={result.get('worst_val')}",
        f"- max/median: {result.get('ratio')}  (median={result.get('median')})",
        f"- real_mean: {result.get('real_mean')}  pad_mean: {result.get('pad_mean')}",
        f"- scalar_loss: {result.get('scalar_loss')}",
        "",
        "## 原因",
        "",
    ]
    for r in result.get("reasons") or []:
        lines.append(f"- {r}")
    lines += ["", "## 前 16 维 FM-MSE（batch×chunk 均值）", "", "| dim | name | loss |", "|---:|---|---:|"]
    for i, v in enumerate(dims):
        n = names[i] if i < len(names) else f"d{i}"
        mark = "  ← worst" if i == result.get("worst_dim") else ""
        lines.append(f"| {i} | {n} | {v}{mark} |")
    lines += [
        "",
        "提醒通道这次没接。看本文件和同目录 `action_dim_abort.json` / `loss_probe.jsonl`。",
        "常见根因：`norm_stats` 与 `delta_action_dims` 掩码不一致，或动作专家没热加载。",
        "",
    ]
    return "\n".join(lines)


def _write_action_dim_abort_report(checkpoint_dir: str, result: dict) -> str:
    """写 json + md，返回 md 路径。"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    base = os.path.join(checkpoint_dir, "action_dim_abort")
    json_path = base + ".json"
    md_path = base + ".md"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_format_action_dim_abort_md(result))
    return md_path


def _guard_action_dims(checkpoint_dir, step, mat, scalar_loss, is_main, device) -> None:
    """前 100 步盯全部 action dim；异常写 report 并停训。PI05_ACTION_DIM_GUARD=0 可关。"""
    if os.environ.get("PI05_ACTION_DIM_GUARD", "1").strip().lower() in ("0", "false", "off", "no"):
        return
    result = _evaluate_action_dims(mat, step=int(step), scalar_loss=scalar_loss)
    abort = bool(result["abort"])
    try:
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([1 if abort else 0], device=device, dtype=torch.int32)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            abort = bool(int(flag.item()))
    except Exception:  # noqa: BLE001 - 集合失败时仍用本 rank 的判断
        pass
    if not abort:
        return
    if is_main:
        try:
            md = _write_action_dim_abort_report(checkpoint_dir, result)
            logging.error("ACTION-DIM ABORT report=%s", md)
        except Exception as e:  # noqa: BLE001
            logging.error("ACTION-DIM ABORT 写 report 失败：%s", e)
        try:
            _write_loss_probe(checkpoint_dir, step, mat)
        except Exception:  # noqa: BLE001
            pass
    msg = "; ".join(result.get("reasons") or ["action-dim guard abort"])
    logging.error("ACTION-DIM ABORT step=%s %s", step, msg)
    raise ActionDimAbort(f"step={step} {msg}  (report: {checkpoint_dir}/action_dim_abort.md)")


def load_pytorch_weight_path(model, weight_path: str, device: str | None = None) -> None:
    """Load ``model.safetensors`` from a π0.5 weight dir.

    ``pi05_fixed`` has a tied embed_tokens key under ``paligemma.model.language_model``
    that current HF PaliGemma does not register; match eval and load non-strict.
    """
    raw = _unwrap_model(model)
    model_path = os.path.join(weight_path, "model.safetensors")
    try:
        if device is not None:
            missing, unexpected = safetensors.torch.load_model(raw, model_path, strict=False, device=device)
        else:
            missing, unexpected = safetensors.torch.load_model(raw, model_path, strict=False)
        logging.info(
            "Weight load (strict=False): missing=%s unexpected=%s",
            len(missing or []),
            len(unexpected or []),
        )
        if missing:
            logging.warning("Missing weight keys (first 12): %s", list(missing)[:12])
        if unexpected:
            logging.info("Unexpected weight keys (first 8): %s", list(unexpected)[:8])
        return
    except TypeError:
        safetensors.torch.load_model(raw, model_path, strict=False)
        return
    except Exception:
        logging.warning("safetensors.load_model failed; remapping keys from %s", model_path, exc_info=True)

    state = safetensors.torch.load_file(model_path)
    model_state = raw.state_dict()
    remapped = {}
    skipped = 0
    for key, tensor in state.items():
        target = key
        if target not in model_state:
            alt = target.replace(".model.language_model.", ".language_model.")
            if alt in model_state:
                target = alt
        if target in model_state and tuple(model_state[target].shape) == tuple(tensor.shape):
            remapped[target] = tensor
        else:
            skipped += 1
    missing, unexpected = raw.load_state_dict(remapped, strict=False)
    logging.info(
        "Remapped weight load: used=%s skipped=%s missing=%s unexpected=%s",
        len(remapped),
        skipped,
        len(missing),
        len(unexpected),
    )


def _unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def param_is_trainable(name: str, mode: str) -> bool:
    mode = (mode or "all").lower()
    if mode == "all":
        return True
    n = name.replace("module.", "")
    is_paligemma = "paligemma_with_expert.paligemma" in n
    is_action = (
        "gemma_expert" in n
        or "action_in_proj" in n
        or "action_out_proj" in n
        or n.startswith("time_mlp")
        or ".time_mlp" in n
        or n.startswith("state_proj")
        or ".state_proj" in n
        or "action_time_mlp" in n
    )
    if mode == "paligemma":
        return is_paligemma
    if mode == "action":
        return is_action
    raise ValueError(f"unknown trainable_modules={mode!r}; expected all|paligemma|action")


def apply_trainable_modules(model, mode: str) -> tuple[int, int]:
    raw = _unwrap_model(model)
    n_on = n_off = 0
    for name, param in raw.named_parameters():
        train = param_is_trainable(name, mode)
        param.requires_grad = train
        if train:
            n_on += param.numel()
        else:
            n_off += param.numel()
    return n_on, n_off


def build_optimizer(model, config: _config.TrainConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError(f"no trainable parameters for trainable_modules={config.trainable_modules}")
    return torch.optim.AdamW(
        params,
        lr=config.lr_schedule.peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )


def resolve_trainable_mode(config: _config.TrainConfig, global_step: int, steps_per_epoch: int | None) -> tuple[str, int | None]:
    stage2_step = None
    if config.stage2_trainable_modules is not None:
        if config.stage2_start_step is not None:
            stage2_step = int(config.stage2_start_step)
        elif config.stage2_start_epoch is not None:
            spe = int(steps_per_epoch or 0)
            if spe < 1:
                raise ValueError("stage2_start_epoch needs a resolved epoch schedule (num_epochs + dataset length)")
            stage2_step = int(config.stage2_start_epoch) * spe
        else:
            raise ValueError("stage2_trainable_modules requires stage2_start_step or stage2_start_epoch")
    if stage2_step is not None and global_step >= stage2_step:
        return str(config.stage2_trainable_modules), stage2_step
    return str(config.trainable_modules), stage2_step


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    config = _schedule.resolve_epoch_schedule(config)
    steps_per_epoch = _schedule.steps_per_epoch_from_config(config)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize local train.log / hardware artifacts (main process only)
    resource_monitor = None
    loss_steps: list[int] = []
    loss_values: list[float] = []
    if is_main:
        _train_log.init_train_log(
            config.checkpoint_dir,
            config,
            append=resuming,
            hardware_extra={
                "framework": "pytorch",
                "world_size": int(torch.distributed.get_world_size() if use_ddp else 1),
                "local_rank": local_rank,
                "device": str(device),
                "exp_name": config.exp_name,
                "config_name": config.name,
            },
        )
        try:
            resource_monitor = _train_log.start_resource_monitor(config.checkpoint_dir)
        except Exception:
            logging.exception("Failed to start resource monitor")
        if resuming:
            loss_steps, loss_values = _train_log.load_train_loss_history(config.checkpoint_dir)

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        load_pytorch_weight_path(
            model,
            config.pytorch_weight_path,
            device=str(device),
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    global_step = 0
    if resuming:
        latest_step = get_latest_checkpoint_step(config.checkpoint_dir)
        if latest_step is not None:
            global_step = int(latest_step)

    trainable_mode, stage2_step = resolve_trainable_mode(config, global_step, steps_per_epoch)
    n_on, n_off = apply_trainable_modules(model, trainable_mode)
    logging.info(
        f"trainable_modules={trainable_mode} params_on={n_on} params_off={n_off} "
        f"stage2={config.stage2_trainable_modules} stage2_step={stage2_step}"
    )
    optim = build_optimizer(model, config)
    stage_switched = bool(
        config.stage2_trainable_modules is not None and trainable_mode == str(config.stage2_trainable_modules)
    )

    if resuming:
        global_step = load_checkpoint(
            model, optim, config.checkpoint_dir, device, total_steps=config.num_train_steps
        )
        logging.info(f"Resumed training from step {global_step}")
        trainable_mode, stage2_step = resolve_trainable_mode(config, global_step, steps_per_epoch)
        n_on, n_off = apply_trainable_modules(model, trainable_mode)
        stage_switched = bool(
            config.stage2_trainable_modules is not None and trainable_mode == str(config.stage2_trainable_modules)
        )
        logging.info(f"Post-resume freeze trainable_modules={trainable_mode} on={n_on} off={n_off}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")
        logging.info(
            f"Freeze schedule: trainable={config.trainable_modules} "
            f"stage2={config.stage2_trainable_modules} stage2_step={stage2_step}"
        )

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device)  # noqa: PLW2901

            if (
                stage2_step is not None
                and not stage_switched
                and global_step >= stage2_step
                and config.stage2_trainable_modules is not None
            ):
                if use_ddp and dist.is_initialized():
                    dist.barrier()
                n_on, n_off = apply_trainable_modules(model, config.stage2_trainable_modules)
                optim = build_optimizer(model, config)
                stage_switched = True
                if is_main:
                    logging.info(
                        f"Stage2 switch at step={global_step}: trainable={config.stage2_trainable_modules} "
                        f"params_on={n_on} params_off={n_off}"
                    )

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            losses = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            if isinstance(losses, list | tuple):
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # per-joint × per-timestep 探针：不进反传，只取 batch 均值
            fm_probe = None
            if isinstance(losses, torch.Tensor) and losses.dim() == 3:
                with torch.no_grad():
                    fm_probe = losses.detach().float().mean(dim=0).cpu().numpy()  # [H, D]
            if fm_probe is not None:
                _guard_action_dims(
                    checkpoint_dir=config.checkpoint_dir,
                    step=global_step,
                    mat=fm_probe,
                    scalar_loss=float(loss.detach()),
                    is_main=is_main,
                    device=device,
                )

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                info = {
                    "loss": loss.item(),
                    "learning_rate": optim.param_groups[0]["lr"],
                    "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                }
                if fm_probe is not None:
                    info["fm_probe"] = fm_probe
                infos.append(info)

            if is_main and (global_step % config.log_interval == 0) and infos:
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    _train_log.format_metrics_line(
                        global_step,
                        loss=avg_loss,
                        grad_norm=avg_grad_norm,
                        lr=avg_lr,
                        update_s=elapsed / max(1, config.log_interval),
                        epoch=(global_step / steps_per_epoch) if steps_per_epoch else None,
                    )
                )
                # Keep a human-readable summary as well.
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                probe_mats = [i["fm_probe"] for i in infos if "fm_probe" in i]
                if probe_mats:
                    _write_loss_probe(config.checkpoint_dir, global_step, sum(probe_mats) / len(probe_mats))
                while loss_steps and loss_steps[-1] >= int(global_step):
                    loss_steps.pop()
                    loss_values.pop()
                loss_steps.append(int(global_step))
                loss_values.append(float(avg_loss))
                _train_log.save_train_loss_plot(config.checkpoint_dir, loss_steps, loss_values)

                # Log to wandb
                if config.wandb_enabled:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    if is_main:
        if loss_steps:
            _train_log.save_train_loss_plot(config.checkpoint_dir, loss_steps, loss_values)
        if resource_monitor is not None:
            try:
                resource_monitor.stop()
            except Exception:
                logging.exception("Failed to stop resource monitor")
        logging.info("End of training")

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
