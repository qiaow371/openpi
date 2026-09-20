from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
import shutil
from pathlib import Path
from typing import Protocol

from etils import epath
import jax
try:
    import orbax.checkpoint as ocp
except Exception:
    ocp = None  # type: ignore[assignment]
try:
    import orbax.checkpoint.future as future
except Exception:
    future = None  # type: ignore[assignment]

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.train_log as _train_log
import openpi.training.utils as training_utils


def asset_checkpoint_relpath(asset_id: str) -> str:
    """Map ``asset_id`` to a relative path that stays under an assets directory.

    ``asset_id="."`` means use the assets directory itself (no subfolder) — the
    usual setup when ``norm_stats.json`` sits on the dataset root and YAML sets
    ``data.assets.assets_dir`` to that root.

    Absolute ``asset_id`` values (often a mistaken copy of ``repo_id``) are
    reduced to a single path segment so joins never discard the left-hand side.
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


def _ensure_relative_link_or_copy(src: Path, dst: Path) -> None:
    """Point ``dst`` at ``src`` via relative symlink; fall back to copy on unsupported FS."""
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        # Empty dir left by a failed previous finalize — remove and replace.
        try:
            next(dst.iterdir())
        except StopIteration:
            dst.rmdir()
        else:
            # Non-empty existing dir: leave as-is (already materialized).
            return
    elif dst.exists():
        return

    rel_target = os.path.relpath(src, dst.parent)
    try:
        os.symlink(rel_target, dst)
    except OSError:
        logging.warning("Symlink unsupported for %s -> %s; copying instead", dst, src)
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=True)
        else:
            shutil.copy2(src, dst)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _move_under_pretrained(step_dir: Path, name: str, pretrained: Path) -> None:
    """Move Orbax ``{step}/{name}`` into ``pretrained_model/{name}`` (real data).

    Leave ``{step}/{name}`` as a relative symlink back to ``pretrained_model/{name}``
    so Orbax resume still finds the item without duplicating data.
    """
    src = step_dir / name
    dst = pretrained / name
    if not src.exists() and not dst.exists():
        logging.warning("Missing %s under %s (and not in pretrained_model)", name, step_dir)
        return

    # Already relocated in a previous finalize.
    if src.is_symlink():
        return

    pretrained.mkdir(parents=True, exist_ok=True)

    if src.exists() and not src.is_symlink():
        if dst.exists():
            _remove_path(dst)
        shutil.move(str(src), str(dst))
    elif not dst.exists():
        logging.warning("Expected %s after Orbax save, missing at %s", name, src)
        return

    # Orbax looks under step/{name}; keep a symlink only (no second copy).
    if src.exists() and not src.is_symlink():
        _remove_path(src)
    if not src.exists():
        _ensure_relative_link_or_copy(dst, src)


def initialize_checkpoint_dir(
    run_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    max_to_keep: int = 10,
    should_keep_fn=None,
) -> tuple[ocp.CheckpointManager, bool]:
    """Initialize Orbax manager under ``{run_dir}/checkpoints`` (ACT-aligned layout)."""
    run_dir = epath.Path(run_dir).resolve()
    checkpoints_dir = run_dir / _train_log.CHECKPOINTS_DIR
    resuming = False
    if run_dir.exists():
        if overwrite:
            run_dir.rmtree()
            run_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {run_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {run_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoints_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max(1, int(max_to_keep)),
            keep_period=None if should_keep_fn is not None else keep_period,
            should_keep_fn=should_keep_fn,
            create=False,
            # Wait until the new step is fully on disk before deleting older
            # ones. Async + max_to_keep=1 was deleting step N while N+1 was
            # still in a tmp dir; an OOM then left zero usable checkpoints.
            enable_async_checkpointing=False,
            enable_background_delete=False,
            cleanup_tmp_directories=True,
            todelete_subdir=".orbax_deleted",
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    include_train_state: bool = True,
):
    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None:
            out_dir = Path(directory)
            _normalize.save(out_dir, norm_stats)
            logging.info("Saved norm stats to %s", out_dir / "norm_stats.json")

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "params": {"params": params},
    }
    if include_train_state:
        items["train_state"] = train_state
        logging.info("Saving full checkpoint (params + train_state) at step %s", step)
    else:
        logging.info("Saving params-only checkpoint at step %s (no optimizer state)", step)
    checkpoint_manager.save(step, items)


def _step_has_train_state(checkpoint_manager: ocp.CheckpointManager, step: int) -> bool:
    return (Path(checkpoint_manager.directory) / str(int(step)) / "train_state").exists()


def latest_full_step(checkpoint_manager: ocp.CheckpointManager) -> int | None:
    """Latest Orbax step that contains ``train_state`` (needed for ``--resume``)."""
    steps = sorted(int(s) for s in checkpoint_manager.all_steps())
    for step in reversed(steps):
        if _step_has_train_state(checkpoint_manager, step):
            return step
    return None


def finalize_step_checkpoint(
    run_dir: epath.Path | str,
    step: int,
    *,
    total_steps: int,
    config: object | None = None,
) -> Path:
    """
    After Orbax writes under ``checkpoints/{step}/``, move weights into
    ``pretrained_model/params``. Keep ``assets/`` at the step root.

    Layout:
      checkpoints/{step}/
        assets/norm_stats.json
        train_state/
        params -> pretrained_model/params
        pretrained_model/
          params/
          config.json
          train_config.json
      checkpoints/last -> {step}
    """
    # Use pathlib.Path for FS ops: etils PosixGPath has no is_symlink()/is_file().
    run_dir = Path(run_dir)
    checkpoints_dir = run_dir / _train_log.CHECKPOINTS_DIR
    step_dir = checkpoints_dir / str(int(step))
    if not step_dir.exists():
        raise FileNotFoundError(f"Expected Orbax step directory missing: {step_dir}")

    pretrained = step_dir / _train_log.PRETRAINED_MODEL_DIR
    pretrained.mkdir(parents=True, exist_ok=True)

    _move_under_pretrained(step_dir, "params", pretrained)
    nested_assets = pretrained / "assets"
    if nested_assets.exists() or nested_assets.is_symlink():
        _remove_path(nested_assets)

    if config is not None:
        try:
            _train_log.save_policy_config(pretrained, config)
            _train_log.save_train_config(pretrained, config)
        except Exception:
            logging.exception("Failed to write config files into %s", pretrained)

    padded = _train_log.get_step_identifier(step, total_steps)
    public_step_dir = step_dir
    if padded != str(int(step)):
        alias = checkpoints_dir / padded
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        if not alias.exists():
            try:
                os.symlink(str(int(step)), alias)
            except OSError:
                logging.warning("Could not create padded step alias %s -> %s", alias, step)
        if alias.exists():
            public_step_dir = alias

    _train_log.update_checkpoint_link(public_step_dir, _train_log.LAST_CHECKPOINT_LINK)
    logging.info("Finalized checkpoint layout: %s", pretrained)
    return pretrained


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    if step is None:
        step = latest_full_step(checkpoint_manager)
        if step is None:
            raise FileNotFoundError(
                "No checkpoint with train_state found; cannot --resume. "
                "Params-only checkpoints are for inference, not optimizer resume. "
                "Train until the first full save (save_full_interval) or disable "
                "save_full_every_epochs."
            )
        latest = max((int(s) for s in checkpoint_manager.all_steps()), default=None)
        if latest is not None and int(step) != int(latest):
            logging.warning(
                "Latest checkpoint is params-only (step %s); resuming from full "
                "train_state at step %s. Intermediate params-only steps will be retrained.",
                latest,
                step,
            )
    elif not _step_has_train_state(checkpoint_manager, step):
        raise FileNotFoundError(
            f"Checkpoint step {step} has no train_state (params-only); cannot restore optimizer."
        )

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str | None = None) -> dict[str, _normalize.NormStats] | None:
    """Load norm stats from checkpoint ``assets/`` (flat) or legacy ``assets/{asset_id}/``."""
    assets_dir = Path(assets_dir)
    candidates = [assets_dir]
    if asset_id:
        rel = asset_checkpoint_relpath(asset_id)
        if rel:
            candidates.extend([assets_dir / rel, assets_dir / Path(asset_id).name])
        elif Path(asset_id).is_absolute():
            candidates.append(Path(asset_id))
    last_error: Exception | None = None
    for norm_stats_dir in candidates:
        try:
            norm_stats = _normalize.load(norm_stats_dir)
            logging.info("Loaded norm stats from %s", norm_stats_dir)
            return norm_stats
        except FileNotFoundError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return None


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


_ocp_base = ocp.AsyncCheckpointHandler if ocp is not None else object
class CallbackHandler(_ocp_base):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


_noop = lambda cls: cls
_cb_base = ocp.args.CheckpointArgs if ocp else object

@dataclasses.dataclass
class CallbackSave(_cb_base):
    callback: Callback


class CallbackRestore(_cb_base): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
