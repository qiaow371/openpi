from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
try:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
except ModuleNotFoundError:
    import lerobot.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def inspect_subtask_sidecar(repo_id: str | None) -> dict:
    """Coverage of AgiBot-style ``action_config`` sidecar next to a LeRobot repo."""
    out: dict = {
        "sidecar": None,
        "annotated_episodes": 0,
        "total_episodes": None,
        "spans": 0,
    }
    if not repo_id or repo_id == "fake":
        return out
    root = pathlib.Path(repo_id)
    meta = root / "meta"
    if not meta.is_dir():
        return out
    info = meta / "info.json"
    if info.is_file():
        import json

        try:
            out["total_episodes"] = json.loads(info.read_text()).get("total_episodes")
        except Exception:
            pass
    sidecar = None
    for name in ("episodes_detailed_task.jsonl", "subtask_spans.jsonl"):
        cand = meta / name
        if cand.is_file():
            sidecar = cand
            break
    if sidecar is None:
        return out
    out["sidecar"] = str(sidecar)
    import json

    eps: set[int] = set()
    n_spans = 0
    with sidecar.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "action_config" in obj:
                ep = int(obj.get("episode_index", obj.get("episode_id", 0)))
                eps.add(ep)
                n_spans += len(obj["action_config"])
            else:
                eps.add(int(obj["episode_index"]))
                n_spans += 1
    out["annotated_episodes"] = len(eps)
    out["spans"] = n_spans
    return out


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if getattr(data_config, "prompt_from_subtask", False):
        root = pathlib.Path(dataset_meta.root) / "meta"
        spans_file = None
        for name in ("episodes_detailed_task.jsonl", "subtask_spans.jsonl"):
            cand = root / name
            if cand.is_file():
                spans_file = cand
                break
        if spans_file is not None:
            cov = inspect_subtask_sidecar(str(dataset_meta.root))
            logging.info(
                "Loading AgiBot-style subtask prompts from %s (%s/%s episodes, %s spans)",
                spans_file,
                cov["annotated_episodes"],
                cov["total_episodes"],
                cov["spans"],
            )
            dataset = TransformedDataset(dataset, [_transforms.PromptFromSubtaskSpans.from_jsonl(spans_file)])
        else:
            logging.warning(
                "data.prompt_from_subtask=true but meta/episodes_detailed_task.jsonl missing; using global task only"
            )

    if getattr(data_config, "subtask_ce", False):
        root = pathlib.Path(dataset_meta.root) / "meta"
        spans_file = None
        for name in ("episodes_detailed_task.jsonl", "subtask_spans.jsonl"):
            cand = root / name
            if cand.is_file():
                spans_file = cand
                break
        if spans_file is not None:
            cov = inspect_subtask_sidecar(str(dataset_meta.root))
            logging.info(
                "Attaching subtask CE labels from %s (%s/%s episodes, %s spans); action prompt stays task",
                spans_file,
                cov["annotated_episodes"],
                cov["total_episodes"],
                cov["spans"],
            )
            dataset = TransformedDataset(dataset, [_transforms.AttachSubtaskFromSpans.from_jsonl(spans_file)])
        else:
            logging.warning(
                "data.subtask_ce=true but meta/episodes_detailed_task.jsonl missing; CE labels will be empty"
            )

    # Debug log: keep per-frame subtask text for detailed_loss/subtask/* even when
    # it is not the action prompt and CE is off. Does not overwrite ``prompt``.
    if not getattr(data_config, "subtask_ce", False):
        root = pathlib.Path(dataset_meta.root) / "meta"
        spans_file = None
        for name in ("episodes_detailed_task.jsonl", "subtask_spans.jsonl"):
            cand = root / name
            if cand.is_file():
                spans_file = cand
                break
        if spans_file is not None:
            dataset = TransformedDataset(dataset, [_transforms.AttachSubtaskFromSpans.from_jsonl(spans_file)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


class FilterByEpisode:
    """Keep only samples whose episode_index is in ``allow`` (used for offline val)."""

    def __init__(self, dataset, allow: set[int]):
        self._dataset = dataset
        self._idxs: list[int] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            ep = sample.get("episode_index", sample.get("episode"))
            if ep is None:
                continue
            ep_i = int(ep.item()) if hasattr(ep, "item") else int(ep)
            if ep_i in allow:
                self._idxs.append(i)
        if not self._idxs:
            raise ValueError(f"FilterByEpisode: 0 frames in allowlist size={len(allow)}")

    def __getitem__(self, index):
        return self._dataset[self._idxs[index]]

    def __len__(self) -> int:
        return len(self._idxs)


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
    episode_allowlist: set[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        episode_allowlist=episode_allowlist,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    episode_allowlist: set[int] | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    if episode_allowlist is not None:
        dataset = FilterByEpisode(dataset, episode_allowlist)
        logging.info("FilterByEpisode keep %s frames from %s episodes", len(dataset), len(episode_allowlist))
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _drop_non_numeric(tree):
    """torch.as_tensor cannot take numpy.str_ / object leaves (e.g. leftover subtask text)."""
    if isinstance(tree, dict):
        out = {}
        for key, value in tree.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in ("U", "S", "O"):
                continue
            out[key] = _drop_non_numeric(value)
        return out
    return tree


_LOSS_CAT_MAX = 96  # utf-8 bytes; closed-set subtask phrases fit, long task prompts do not


def _item_cat_str(item, key: str) -> str:
    if not isinstance(item, dict):
        return ""
    v = item.get(key, "")
    if v is None:
        return ""
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            v = str(v)
    return str(v).strip()


def _encode_loss_cat_batch(strs: list[str], max_len: int = _LOSS_CAT_MAX) -> np.ndarray:
    arr = np.zeros((len(strs), max_len), dtype=np.uint8)
    for i, s in enumerate(strs):
        b = (s or "").encode("utf-8", errors="replace")[:max_len]
        if b:
            arr[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    return arr


def _decode_loss_cat_batch(tensor) -> list[str] | None:
    if tensor is None:
        return None
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    arr = np.asarray(tensor)
    if arr.ndim != 2:
        return None
    out = []
    for row in arr:
        text = bytes(int(x) for x in row.tolist()).split(b"\x00", 1)[0]
        out.append(text.decode("utf-8", errors="replace").strip())
    return out


def _pack_loss_cats(items) -> dict:
    """Keep prompt/subtask through collate as uint8 so debug log can group FM by category."""
    packed = {}
    prompts = [_item_cat_str(it, "prompt") for it in items]
    subtasks = [_item_cat_str(it, "subtask") for it in items]
    if any(prompts):
        packed["loss_cat_prompt"] = _encode_loss_cat_batch(prompts)
    if any(subtasks):
        packed["loss_cat_subtask"] = _encode_loss_cat_batch(subtasks)
    return packed


def _unpack_loss_cats(batch, data_config) -> dict:
    repo = ""
    if data_config is not None:
        repo = str(getattr(data_config, "repo_id", "") or "")
    repo = os.path.basename(str(repo).rstrip("/")) or "dataset"
    cats = {"dataset": repo}
    if isinstance(batch, dict):
        decoded = _decode_loss_cat_batch(batch.get("loss_cat_subtask"))
        if decoded:
            cats["subtask"] = decoded
        decoded_p = _decode_loss_cat_batch(batch.get("loss_cat_prompt"))
        if decoded_p:
            cats["prompt"] = decoded_p
    return cats


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    batched = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)
    batched = _drop_non_numeric(batched)
    if isinstance(batched, dict):
        batched.update(_pack_loss_cats(items))
    return batched


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            self._debug_loss_cats = _unpack_loss_cats(batch, self._data_config)
            yield _model.Observation.from_dict(batch), batch["actions"]
