"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Location of precomputed assets (e.g. ``norm_stats.json``) for the data pipeline.

    Norm stats are **not** computed during training. Precompute them (e.g. with
    ``scripts/compute_norm_stats.py``) onto the dataset root, then point here so
    training only **loads** them and copies into the checkpoint ``assets/`` for
    inference.

    Local dataset (file sits at ``{repo}/norm_stats.json``)::

        AssetsConfig(assets_dir="/path/to/lerobot_dataset", asset_id=".")

    Or reload from a base checkpoint (e.g. Trossen / ALOHA)::

        AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        )
    """

    # Directory that contains norm_stats (or a subdirectory named by asset_id).
    # If unset, falls back to TrainConfig.assets_dirs (``assets_base_dir`` only).
    assets_dir: str | None = None

    # Subfolder under assets_dir. Use ``"."`` when norm_stats.json is directly
    # under assets_dir (typical local dataset layout). If unset, repo_id is used.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # If true and ``<repo>/meta/subtask_spans.jsonl`` exists, overwrite prompt
    # per frame with the matching subtask (π0.5 ˆℓ). Misses keep the global task.
    prompt_from_subtask: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                pi05_inputs = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                ]
                if getattr(model_config, "use_depth_encoder", False):
                    pi05_inputs.append(_transforms.ResizeDepths(224, 224))
                pi05_inputs.extend(
                    [
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ]
                )
                return _transforms.Group(inputs=pi05_inputs)
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        from openpi.training.checkpoints import asset_checkpoint_relpath

        # Precomputed norm_stats.json is loaded here (not computed during training).
        # asset_id="." means the file sits directly under assets_dir.
        candidates: list[epath.Path] = [epath.Path(assets_dir)]
        if asset_id:
            rel = asset_checkpoint_relpath(asset_id)
            if rel:
                candidates.append(epath.Path(assets_dir) / rel)
            if pathlib.Path(asset_id).is_absolute():
                candidates.append(epath.Path(asset_id))

        for data_assets_dir in candidates:
            try:
                norm_stats = _normalize.load(_download.maybe_download(str(data_assets_dir)))
                logging.info(f"Loaded norm stats from {data_assets_dir}")
                return norm_stats
            except FileNotFoundError:
                continue
        logging.info(f"Norm stats not found under {assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, load prompt from LeRobot meta/tasks.jsonl via each sample's task_index
    # (see data_loader.create_torch_dataset → PromptFromLeRobotTask).
    # YAML: data.prompt_from_task: true
    prompt_from_task: bool = True
    # YAML: data.prompt_from_subtask: true  （需要 meta/episodes_detailed_task.jsonl）
    prompt_from_subtask: bool = False
    # YAML: data.append_modality_prompt: true
    # 在 prefix 里、每个 DEPTH/FT 模态的 token 前面插入对应语言 token（不是拼进 Task）。
    append_modality_prompt: bool = False
    # YAML: data.modality_prompt_tags: {cam_left_depth: "DEPTH WRIST LEFT: "}
    modality_prompt_tags: dict[str, str] = dataclasses.field(default_factory=dict)
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
    # make_bool_mask(*delta_action_dims). Aloha14D:(6,-1,6,-1) TongBot16D:(7,7,-1,-1)
    # YAML: data.delta_action_dims: [7, 7, -1, -1]
    delta_action_dims: Sequence[int] = (6, -1, 6, -1)

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"base_0_rgb": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        extra_inputs = []
        has_sidecar_depth = any(
            isinstance(t, aloha_policy.LoadSidecarDepthPNGs) for t in self.repack_transforms.inputs
        )
        if has_sidecar_depth:
            # depth 走 SigLIP，不建独立 DepthEncoder
            extra_inputs.append(aloha_policy.ProcessDepths())
            extra_inputs.append(aloha_policy.DepthsAsSiglipImages())
        elif getattr(model_config, "use_depth_encoder", False):
            extra_inputs.append(aloha_policy.ProcessDepths())
        if getattr(model_config, "use_force6d_encoder", False):
            extra_inputs.append(aloha_policy.ProcessForce6D())
        data_transforms = _transforms.Group(
            inputs=[*extra_inputs, aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(
                *[int(x) for x in self.delta_action_dims]
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        repack_transforms = self.repack_transforms
        if self.repo_id is not tyro.MISSING and self.repo_id:
            patched = []
            changed = False
            for t in repack_transforms.inputs:
                if isinstance(t, aloha_policy.LoadSidecarDepthPNGs):
                    t = dataclasses.replace(t, dataset_root=str(self.repo_id))
                    changed = True
                patched.append(t)
            if changed:
                repack_transforms = dataclasses.replace(repack_transforms, inputs=tuple(patched))

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
            prompt_from_subtask=self.prompt_from_subtask,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaPoseDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    use_pose_state_inputs: bool = False
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"base_0_rgb": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.PoseDeltaActions(delta_action_mask)],
                outputs=[_transforms.PoseAbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPikaPoseDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    use_pose_state_inputs: bool = False
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = False

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"base_0_rgb": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi), _transforms.RelativePoseStateInputs(), _transforms.PoseDeltaActions(delta_action_mask)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi), _transforms.PoseAbsoluteActions(delta_action_mask)],
        )
        # if self.use_delta_joint_actions:
        #     delta_action_mask = _transforms.make_bool_mask(9, -1, 9, -1)
        #     data_transforms = data_transforms.push(
        #         inputs=[],
        #         outputs=[],
        #     )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Fallback assets root only when data.assets.assets_dir is unset.
    # Preferred flow: precompute norm_stats.json on the dataset, point
    # data.assets.assets_dir at that dataset (usually = repo_id).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    # When ``num_epochs`` is set, this is overwritten at startup:
    # ``num_train_steps = num_epochs * (len(dataset) // batch_size)``.
    num_train_steps: int = 30_000
    # Optional standard-epoch training budget (1 epoch = one full dataset pass).
    # If set, takes priority over ``num_train_steps``.
    num_epochs: int | None = None
    # When set together with ``num_epochs``, converts to
    # ``save_interval = save_every_epochs * steps_per_epoch``.
    save_every_epochs: int | None = None
    # Save full train_state (weights + Adam m/v) every N epochs. Unset = every
    # ``save_interval`` still writes a full checkpoint (old behavior).
    # When set, regular saves are params-only (inference); ``--resume`` needs the
    # latest full train_state and will skip newer params-only steps.
    save_full_every_epochs: int | None = None

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # How often (in steps) to save full train_state. None = always include it.
    # Resolved from ``save_full_every_epochs`` when that field is set.
    save_full_interval: int | None = None
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # PyTorch only: which π0.5 modules receive flow-matching gradients.
    # ``paligemma`` = VLM prefix (vision + language); ``action`` = Gemma expert + action/time heads.
    trainable_modules: Literal["all", "paligemma", "action"] = "all"
    # Optional second freeze schedule. Switch at the start of ``stage2_start_epoch`` (0-based epoch count).
    stage2_trainable_modules: Literal["all", "paligemma", "action"] | None = None
    stage2_start_epoch: int | None = None
    # If set, wins over ``stage2_start_epoch`` (switch after this many completed steps).
    stage2_start_step: int | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Assets directory: ``{assets_base_dir}``."""
        return pathlib.Path(self.assets_base_dir).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Run output directory: ``{checkpoint_base_dir}/{exp_name}``."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.exp_name).resolve()

    @property
    def checkpoints_dir(self) -> pathlib.Path:
        """Step checkpoints live under ``{checkpoint_dir}/checkpoints/``."""
        return self.checkpoint_dir / "checkpoints"

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# cigai20-ft-depth / TongBot：三路 RGB 永远进网；DEPTH 三选一 HEAD / LEFT WRIST / RIGHT WRIST；FT 独立开关。
# 不改 pi05_aloha（front/left/right 旧数据集）。HEAD = cam_mid。
_TONGBOT_FT_DEPTH_PLACEHOLDER = "/home/agilex/dataset-pika_aloha-joint"
_TONGBOT_RGB_CAMS = {
    "base_0_rgb": "observation.images.cam_mid",
    "left_wrist_0_rgb": "observation.images.cam_left",
    "right_wrist_0_rgb": "observation.images.cam_right",
}
_TONGBOT_DEPTH_BY_CAM: dict[str, dict[str, str]] = {
    "head": {"cam_mid": "observation.depths.cam_mid"},
    "left": {"cam_left": "observation.depths.cam_left"},
    "right": {"cam_right": "observation.depths.cam_right"},
}
_TONGBOT_FORCE6D = {
    "left": "observation.force6d.left",
    "right": "observation.force6d.right",
}
_TONGBOT_PROMPT_TAGS_BY_DEPTH: dict[str, dict[str, str]] = {
    "none": {},
    "head": {"cam_mid_depth": "DEPTH HEAD: "},
    "left": {"cam_left_depth": "DEPTH WRIST LEFT: "},
    "right": {"cam_right_depth": "DEPTH WRIST RIGHT: "},
}
_TONGBOT_PROMPT_TAGS_FT: dict[str, str] = {
    "force6d.left": "FT LEFT: ",
    "force6d.right": "FT RIGHT: ",
}


def _tongbot_modality_prompt_tags(
    depth: Literal["none", "head", "left", "right"],
    use_ft: bool,
) -> dict[str, str]:
    """Prefix labels inserted immediately before each extra modality, not onto Task."""
    tags = dict(_TONGBOT_PROMPT_TAGS_BY_DEPTH[depth])
    if use_ft:
        tags.update(_TONGBOT_PROMPT_TAGS_FT)
    return tags


def _tongbot_modality_repack(
    *,
    depth: Literal["none", "head", "left", "right"],
    use_ft: bool,
    dataset_root: str = _TONGBOT_FT_DEPTH_PLACEHOLDER,
) -> _transforms.Group:
    """Build Repack (+ optional sidecar PNG loader) for one modality combo."""
    structure: dict[str, Any] = {
        "images": dict(_TONGBOT_RGB_CAMS),
        "state": "observation.state",
        "actions": "action",
    }
    inputs: list[Any] = []
    if depth != "none":
        depth_map = dict(_TONGBOT_DEPTH_BY_CAM[depth])
        structure["depths"] = depth_map
        inputs.append(
            aloha_policy.LoadSidecarDepthPNGs(
                dataset_root=dataset_root,
                depth_keys=tuple(depth_map.values()),
            )
        )
    if use_ft:
        structure["force6d"] = dict(_TONGBOT_FORCE6D)
    inputs.append(_transforms.RepackTransform(structure))
    return _transforms.Group(inputs=inputs)


def _tongbot_modality_train_config(
    *,
    name: str,
    depth: Literal["none", "head", "left", "right"],
    use_ft: bool,
) -> TrainConfig:
    tag_map = _tongbot_modality_prompt_tags(depth, use_ft)
    return TrainConfig(
        name=name,
        model=pi0_config.Pi0Config(
            pi05=True,
            use_depth_encoder=False,
            use_force6d_encoder=use_ft,
            use_modality_prompt_tokens=False,
            modality_prompt_tags=tag_map,
        ),
        data=LeRobotAlohaDataConfig(
            repo_id=_TONGBOT_FT_DEPTH_PLACEHOLDER,
            assets=AssetsConfig(
                assets_dir=_TONGBOT_FT_DEPTH_PLACEHOLDER,
                asset_id=".",
            ),
            default_prompt="null",
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            # TongBot 16D: 左7 | 右7 | 左爪 abs | 右爪 abs
            delta_action_dims=(7, 7, -1, -1),
            # YAML data.append_modality_prompt: true 才在 DEPTH/FT 前插入语言 token（RGB 不加）。
            append_modality_prompt=False,
            modality_prompt_tags=tag_map,
            repack_transforms=_tongbot_modality_repack(depth=depth, use_ft=use_ft),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    )


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha_1",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha_1",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.cam_high",
                                "left_wrist_0_rgb": "observation.images.cam_left_wrist",
                                "right_wrist_0_rgb": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.cam_high",
                                "left_wrist_0_rgb": "observation.images.cam_left_wrist",
                                "right_wrist_0_rgb": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"), #, action_dim=14),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/agilex/data/lerobot",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi0_aloha_lora",
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"), #, action_dim=14),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/agilex/data/lerobot",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=80000,
        batch_size=16,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA finetuning
    ),

    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/agilex/dataset-pika_aloha-joint",
            # Precompute: scripts/compute_norm_stats.py → {repo_id}/norm_stats.json
            # YAML data.assets can override these paths.
            assets=AssetsConfig(
                assets_dir="/home/agilex/dataset-pika_aloha-joint",
                asset_id=".",
            ),
            default_prompt="null",
            # 自定义双臂 16D（左8|右8，joint7=夹爪）不要走 Aloha14D 的 pi 空间翻转
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            # [左7关节 delta | 左爪 abs | 右7关节 delta | 右爪 abs]
            delta_action_dims=(7, -1, 7, -1),
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    # DATA_COLLECT cigai20-ft-depth：不要改 pi05_aloha。只开 boolean 不够，必须换 Repack + PNG loader。
    TrainConfig(
        name="pi05_aloha_ft_depth",
        model=pi0_config.Pi0Config(pi05=True, use_depth_encoder=False, use_force6d_encoder=True),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/agilex/dataset-pika_aloha-joint",
            assets=AssetsConfig(
                assets_dir="/home/agilex/dataset-pika_aloha-joint",
                asset_id=".",
            ),
            default_prompt="null",
            adapt_to_pi=False,
            use_delta_joint_actions=True,
            delta_action_dims=(7, -1, 7, -1),
            repack_transforms=_transforms.Group(
                inputs=[
                    aloha_policy.LoadSidecarDepthPNGs(
                        dataset_root="/home/agilex/dataset-pika_aloha-joint",
                    ),
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.cam_mid",
                                "left_wrist_0_rgb": "observation.images.cam_left",
                                "right_wrist_0_rgb": "observation.images.cam_right",
                            },
                            "depths": {
                                "cam_left": "observation.depths.cam_left",
                                "cam_mid": "observation.depths.cam_mid",
                                "cam_right": "observation.depths.cam_right",
                            },
                            "force6d": {
                                "left": "observation.force6d.left",
                                "right": "observation.force6d.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    ),
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    # 本机 cigai20-ft-depth 消融：RGB 三路固定；DEPTH=HEAD|LEFT|RIGHT；FT 开/关。
    _tongbot_modality_train_config(name="pi05_aloha_rgb", depth="none", use_ft=False),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_head", depth="head", use_ft=False),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_left", depth="left", use_ft=False),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_right", depth="right", use_ft=False),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_ft", depth="none", use_ft=True),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_head_ft", depth="head", use_ft=True),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_left_ft", depth="left", use_ft=True),
    _tongbot_modality_train_config(name="pi05_aloha_rgb_depth_right_ft", depth="right", use_ft=True),
    TrainConfig(
        name="pi05_aloha_lora",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="/home/agilex/dataset/lerobot/fold_clothes_mix_lerobot",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.front",
                                "left_wrist_0_rgb": "observation.images.left",
                                "right_wrist_0_rgb": "observation.images.right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA finetuning
        batch_size=16,
    ),

    TrainConfig(
        name="pi05_aloha_pose",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/dataset/lerobot/pika_aloha-aloha-pose6d/",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.headerDepthCamera",
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_aloha_pose_lora",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/dataset/lerobot/pika_aloha-aloha-pose6d/",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.headerDepthCamera",
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA finetuning
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_pika_no_header",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotPikaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/pika_data-lerobot",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_pika_no_header_lora",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotPikaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/pika_data-lerobot",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA finetuning
        batch_size=16,
    ),

    TrainConfig(
        name="pi05_pika_with_header",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotPikaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/dataset/lerobot/pika_aloha-pika-pose6d",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.headerDepthCamera",
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        batch_size=16,
    ),
    TrainConfig(
        name="pi05_pika_with_header_lora",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotPikaPoseDataConfig(
            adapt_to_pi=False,
            repo_id="/home/agilex/dataset/lerobot/pika_aloha-pika-pose6d",
            assets=AssetsConfig(
                # assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                # asset_id="trossen",
            ),
            default_prompt="null",
            # base_config=DataConfig(
            #     local_files_only=True,
            # ),
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "base_0_rgb": "observation.images.headerDepthCamera",
                                "left_wrist_0_rgb": "observation.images.pikaDepthCamera_l",
                                "right_wrist_0_rgb": "observation.images.pikaDepthCamera_r",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=80000,
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,  # Turn off EMA for LoRA finetuning
        batch_size=16,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
