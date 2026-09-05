"""Regression tests for multi-checkpoint keep / params-only vs full saves."""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from openpi.training import checkpoints as _checkpoints
from openpi.training import train_log as _train_log


def _manager(tmp_path: Path, *, max_to_keep: int, keep_period: int | None = None):
    run_dir = tmp_path / "run"
    mngr, resuming = _checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=keep_period,
        overwrite=True,
        resume=False,
        max_to_keep=max_to_keep,
    )
    assert resuming is False
    return mngr, run_dir


def _save(mngr, step: int, *, full: bool) -> None:
    def save_assets(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "norm_stats.json").write_text("{}", encoding="utf-8")

    items = {
        "assets": save_assets,
        "params": {"params": {"w": jnp.ones((8, 8), dtype=jnp.float32) * step}},
    }
    if full:
        items["train_state"] = {
            "step": jnp.asarray(step),
            "opt": jnp.ones((4, 4), dtype=jnp.float32) * step,
        }
    mngr.save(step, items)
    mngr.wait_until_finished()


def _disk_steps(run_dir: Path) -> list[int]:
    ckpt_root = run_dir / _train_log.CHECKPOINTS_DIR
    steps = []
    for child in ckpt_root.iterdir():
        if child.is_symlink():
            continue
        if child.is_dir() and child.name.isdigit():
            steps.append(int(child.name))
    return sorted(steps)


def test_max_to_keep_retains_several_finished_checkpoints(tmp_path: Path):
    mngr, run_dir = _manager(tmp_path, max_to_keep=3, keep_period=None)
    for step in (100, 200, 300, 400):
        _save(mngr, step, full=False)
        _checkpoints.finalize_step_checkpoint(run_dir, step, total_steps=1000)

    steps = _disk_steps(run_dir)
    assert steps == [200, 300, 400], steps
    assert not (run_dir / _train_log.CHECKPOINTS_DIR / "100").exists()
    assert (run_dir / _train_log.CHECKPOINTS_DIR / "200" / "pretrained_model" / "params").exists()


def test_max_to_keep_10_does_not_drop_early_saves(tmp_path: Path):
    mngr, run_dir = _manager(tmp_path, max_to_keep=10, keep_period=None)
    saved = [3008, 6016, 9024, 12032]
    for step in saved:
        _save(mngr, step, full=False)
        _checkpoints.finalize_step_checkpoint(run_dir, step, total_steps=150400)

    assert _disk_steps(run_dir) == saved


def test_keep_period_preserves_full_checkpoint_when_evicting(tmp_path: Path):
    mngr, run_dir = _manager(tmp_path, max_to_keep=1, keep_period=10)
    _save(mngr, 10, full=True)
    _checkpoints.finalize_step_checkpoint(run_dir, 10, total_steps=100)
    _save(mngr, 20, full=False)
    _checkpoints.finalize_step_checkpoint(run_dir, 20, total_steps=100)
    _save(mngr, 30, full=False)
    _checkpoints.finalize_step_checkpoint(run_dir, 30, total_steps=100)

    steps = _disk_steps(run_dir)
    assert 10 in steps, f"full keep_period step was deleted: {steps}"
    assert 30 in steps
    assert _checkpoints.latest_full_step(mngr) == 10


def test_latest_full_step_skips_params_only(tmp_path: Path):
    mngr, _run_dir = _manager(tmp_path, max_to_keep=10, keep_period=None)
    _save(mngr, 2, full=True)
    _save(mngr, 4, full=False)
    _save(mngr, 6, full=False)
    assert _checkpoints.latest_full_step(mngr) == 2
    assert not _checkpoints._step_has_train_state(mngr, 4)
    assert _checkpoints._step_has_train_state(mngr, 2)


def test_incomplete_tmp_does_not_remove_previous_checkpoint(tmp_path: Path):
    mngr, run_dir = _manager(tmp_path, max_to_keep=1, keep_period=None)
    _save(mngr, 3008, full=False)
    _checkpoints.finalize_step_checkpoint(run_dir, 3008, total_steps=150400)
    mngr.close()

    ckpt_root = run_dir / _train_log.CHECKPOINTS_DIR
    tmp = ckpt_root / "6016.orbax-checkpoint-tmp-3"
    tmp.mkdir()
    (tmp / "partial").write_text("interrupted", encoding="utf-8")

    mngr2, resuming = _checkpoints.initialize_checkpoint_dir(
        run_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
        max_to_keep=1,
    )
    assert resuming is True
    assert 3008 in [int(s) for s in mngr2.all_steps()]
    assert (ckpt_root / "3008").is_dir()
    assert not tmp.exists(), "stale tmp from crashed save should be cleaned on resume"
    mngr2.close()


def test_params_and_full_saves_never_share_a_step():
    from openpi.training import schedule as _schedule

    save_interval = 1504
    save_full_interval = 15040
    offset = _schedule.staggered_full_offset(save_interval, save_full_interval)
    assert offset == 752
    collided = []
    for step in range(1, 50000):
        params = _schedule.is_params_save_step(step, save_interval, 0, is_last=False)
        full = _schedule.is_full_save_step(step, save_full_interval, offset, 0)
        if params and full:
            collided.append(step)
    assert collided == []
    assert _schedule.is_params_save_step(4512, save_interval, 3008, is_last=False)
    assert not _schedule.is_full_save_step(4512, save_full_interval, offset, 3008)
    assert _schedule.is_full_save_step(15792, save_full_interval, offset, 3008)
    assert not _schedule.is_params_save_step(15792, save_interval, 3008, is_last=False)
