"""Resolve epoch-based training budgets into step counts."""

from __future__ import annotations

import dataclasses
import logging

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def compute_dataset_length(config: _config.TrainConfig) -> tuple[int, bool]:
    """
    Return ``(num_samples, is_approximate)``.

    Matches training data construction for length purposes. RLDS/DROID length is
    a hardcoded approximation (see ``DroidRldsDataset.__len__``).
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        dataset = _data_loader.create_rlds_dataset(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            shuffle=False,
        )
        return len(dataset), True

    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    return len(dataset), False


def staggered_full_offset(save_interval: int, save_full_interval: int | None) -> int:
    """Put optimizer saves at half a params-interval so the two never share a step."""
    if save_full_interval is None:
        return 0
    interval = int(save_interval)
    if interval <= 1:
        return 0
    return interval // 2


def is_params_save_step(step: int, save_interval: int, start_step: int, *, is_last: bool) -> bool:
    if is_last:
        return True
    return step > start_step and save_interval > 0 and step % int(save_interval) == 0


def is_full_save_step(
    step: int,
    save_full_interval: int | None,
    save_full_offset: int,
    start_step: int,
) -> bool:
    if save_full_interval is None:
        return False
    offset = int(save_full_offset or 0)
    interval = int(save_full_interval)
    if interval <= 0 or step <= start_step or step < offset:
        return False
    return (step - offset) % interval == 0


def steps_per_epoch_from_config(config: _config.TrainConfig) -> int | None:
    """Return resolved steps/epoch when ``num_epochs`` was used, else ``None``."""
    if config.num_epochs is None or config.num_epochs <= 0:
        return None
    return config.num_train_steps // int(config.num_epochs)


def resolve_epoch_schedule(config: _config.TrainConfig) -> _config.TrainConfig:
    """
    If ``num_epochs`` is set, convert to a step budget:

    - ``steps_per_epoch = len(dataset) // batch_size``
    - ``num_train_steps = num_epochs * steps_per_epoch``
    - ``lr_schedule.decay_steps = num_train_steps`` (when the schedule has that field)
    - ``save_interval = save_every_epochs * steps_per_epoch`` (when ``save_every_epochs`` is set)
    - ``save_full_interval = save_full_every_epochs * steps_per_epoch`` (when set)

    Idempotent for already-resolved configs that still carry ``num_epochs``.
    """
    if config.num_epochs is None:
        if config.save_every_epochs is not None:
            logging.warning(
                "save_every_epochs=%s ignored because num_epochs is not set; using save_interval=%s",
                config.save_every_epochs,
                config.save_interval,
            )
        if config.save_full_every_epochs is not None:
            logging.warning(
                "save_full_every_epochs=%s ignored because num_epochs is not set; using save_full_interval=%s",
                config.save_full_every_epochs,
                config.save_full_interval,
            )
        offset = staggered_full_offset(config.save_interval, config.save_full_interval)
        if offset != int(config.save_full_offset or 0):
            return dataclasses.replace(config, save_full_offset=offset)
        return config

    num_epochs = int(config.num_epochs)
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be positive, got {config.num_epochs}")

    num_samples, approximate = compute_dataset_length(config)
    steps_per_epoch = num_samples // config.batch_size
    if steps_per_epoch < 1:
        raise ValueError(
            f"steps_per_epoch < 1: dataset_len={num_samples}, batch_size={config.batch_size}. "
            "Increase dataset size or reduce batch_size."
        )

    num_train_steps = num_epochs * steps_per_epoch
    replace_kwargs: dict = {"num_train_steps": num_train_steps}

    if hasattr(config.lr_schedule, "decay_steps"):
        replace_kwargs["lr_schedule"] = dataclasses.replace(
            config.lr_schedule, decay_steps=num_train_steps
        )

    if config.save_every_epochs is not None:
        save_every = int(config.save_every_epochs)
        if save_every <= 0:
            raise ValueError(f"save_every_epochs must be positive, got {config.save_every_epochs}")
        replace_kwargs["save_interval"] = save_every * steps_per_epoch

    if config.save_full_every_epochs is not None:
        save_full_every = int(config.save_full_every_epochs)
        if save_full_every < 0:
            raise ValueError(
                f"save_full_every_epochs must be >= 0, got {config.save_full_every_epochs}"
            )
        if save_full_every == 0:
            # Params-only forever; --resume still needs an existing full train_state.
            replace_kwargs["save_full_interval"] = 0
            replace_kwargs["save_full_offset"] = 0
        else:
            save_full_interval = save_full_every * steps_per_epoch
            replace_kwargs["save_full_interval"] = save_full_interval
            save_interval = replace_kwargs.get("save_interval", config.save_interval)
            replace_kwargs["save_full_offset"] = staggered_full_offset(
                save_interval, save_full_interval
            )

    save_interval = replace_kwargs.get("save_interval", config.save_interval)
    save_full_interval = replace_kwargs.get("save_full_interval", config.save_full_interval)
    if save_full_interval is not None and "save_full_offset" not in replace_kwargs:
        replace_kwargs["save_full_offset"] = staggered_full_offset(save_interval, save_full_interval)

    resolved = dataclasses.replace(config, **replace_kwargs)
    logging.info(
        "Epoch schedule: num_epochs=%s dataset_len=%s%s batch_size=%s "
        "steps_per_epoch=%s num_train_steps=%s save_interval=%s save_full_interval=%s "
        "save_full_offset=%s keep_period=%s decay_steps=%s",
        num_epochs,
        num_samples,
        " (approximate)" if approximate else "",
        config.batch_size,
        steps_per_epoch,
        resolved.num_train_steps,
        resolved.save_interval,
        resolved.save_full_interval,
        resolved.save_full_offset,
        resolved.keep_period,
        getattr(resolved.lr_schedule, "decay_steps", None),
    )
    return resolved
