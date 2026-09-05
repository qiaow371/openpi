from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
import os
from pathlib import Path
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.train_log as _train_log
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    run_dir: epath.Path | str, *, keep_period: int | None, overwrite: bool, resume: bool
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
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
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
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def finalize_step_checkpoint(
    run_dir: epath.Path | str,
    step: int,
    *,
    total_steps: int,
    config: object | None = None,
) -> epath.Path:
    """
    After Orbax finishes writing ``checkpoints/{step}/{params,assets,train_state}``,
    add ACT-style ``pretrained_model/`` views + ``last`` / padded aliases.

    Layout:
      checkpoints/{step}/
        params/, assets/, train_state/          # Orbax (resume)
        pretrained_model/
          params -> ../params
          assets -> ../assets
          config.json
          train_config.json
      checkpoints/{padded} -> {step}            # when padded != str(step)
      checkpoints/last -> {padded or step}
    """
    run_dir = epath.Path(run_dir)
    checkpoints_dir = run_dir / _train_log.CHECKPOINTS_DIR
    step_dir = checkpoints_dir / str(int(step))
    if not step_dir.exists():
        raise FileNotFoundError(f"Expected Orbax step directory missing: {step_dir}")

    pretrained = step_dir / _train_log.PRETRAINED_MODEL_DIR
    pretrained.mkdir(parents=True, exist_ok=True)

    for name in ("params", "assets"):
        src = step_dir / name
        dst = pretrained / name
        if not src.exists():
            continue
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        elif dst.exists():
            continue
        # Relative symlink so the bundle stays portable within the run dir.
        os.symlink(os.path.relpath(src, pretrained), dst)

    if config is not None:
        try:
            _train_log.save_policy_config(pretrained, config)
            _train_log.save_train_config(pretrained, config)
        except Exception:
            logging.exception("Failed to write config files into %s", pretrained)

    padded = _train_log.get_step_identifier(step, total_steps)
    public_step_dir = Path(step_dir)
    if padded != str(int(step)):
        alias = Path(checkpoints_dir) / padded
        if alias.is_symlink() or alias.is_file():
            alias.unlink()
        if not alias.exists():
            os.symlink(str(int(step)), alias)
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
        rel = _train_log.asset_checkpoint_relpath(asset_id)
        if rel:
            candidates.extend([assets_dir / rel, assets_dir / Path(asset_id).name])
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


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


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
