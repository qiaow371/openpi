#!/usr/bin/env python3
"""Patch openpi-jiangsuan so YAML prompt_from_subtask actually loads the sidecar."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("OPENPI_ROOT", "/nvme1n1/openpi/cigai_train/openpi-jiangsuan"))


def once(path, old, new, label):
    text = path.read_text()
    if new in text and old not in text:
        print("[skip] %s already applied" % label)
        return
    if old not in text:
        raise SystemExit("[fail] %s: needle not found in %s" % (label, path))
    path.write_text(text.replace(old, new, 1))
    print("[ok] %s" % label)


def main() -> None:
    cfg = ROOT / "src/openpi/training/config.py"
    once(
        cfg,
        """    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
""",
        """    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False
    # If true, overwrite prompt per frame from meta/episodes_detailed_task.jsonl.
    prompt_from_subtask: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
""",
        label="DataConfig.prompt_from_subtask",
    )
    once(
        cfg,
        """    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
""",
        """    default_prompt: str | None = None
    prompt_from_task: bool = True
    prompt_from_subtask: bool = False
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True
""",
        label="LeRobotAlohaDataConfig fields",
    )
    once(
        cfg,
        """        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaPoseDataConfig:
""",
        """        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=self.prompt_from_task,
            prompt_from_subtask=self.prompt_from_subtask,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaPoseDataConfig:
""",
        label="LeRobotAlohaDataConfig.create",
    )

    dl = ROOT / "src/openpi/training/data_loader.py"
    once(
        dl,
        "import multiprocessing\nimport os\nimport typing\n",
        "import multiprocessing\nimport os\nimport pathlib\nimport typing\n",
        label="data_loader pathlib",
    )
    once(
        dl,
        """def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    \"\"\"Create a dataset for training.\"\"\"
""",
        """def inspect_subtask_sidecar(repo_id: str | None) -> dict:
    \"\"\"Coverage of AgiBot-style action_config sidecar next to a LeRobot repo.\"\"\"
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
    \"\"\"Create a dataset for training.\"\"\"
""",
        label="inspect_subtask_sidecar",
    )
    once(
        dl,
        """    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset
""",
        """    if data_config.prompt_from_task:
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

    return dataset
""",
        label="create_torch_dataset subtask",
    )

    tr = ROOT / "src/openpi/transforms.py"
    once(
        tr,
        "import dataclasses\nimport math\nimport re\n",
        "import dataclasses\nimport math\nimport pathlib\nimport re\n",
        label="transforms pathlib",
    )
    once(
        tr,
        """        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
""",
        """        return {**data, "prompt": prompt}


def _as_int(value) -> int:
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


@dataclasses.dataclass(frozen=True)
class PromptFromSubtaskSpans(DataTransformFn):
    \"\"\"Per-frame prompt from AgiBot-style action_config sidecar.\"\"\"

    spans: dict[int, tuple[tuple[int, int, str], ...]]

    @classmethod
    def from_jsonl(cls, path: str | pathlib.Path) -> "PromptFromSubtaskSpans":
        import json

        by_ep: dict[int, list[tuple[int, int, str]]] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "action_config" in obj:
                    ep = int(obj.get("episode_index", obj.get("episode_id", 0)))
                    for sl in obj["action_config"]:
                        text = str(sl.get("action_text") or sl.get("subtask") or "")
                        by_ep.setdefault(ep, []).append(
                            (int(sl["start_frame"]), int(sl["end_frame"]), text)
                        )
                    continue
                ep = int(obj["episode_index"])
                text = str(obj.get("action_text") or obj.get("subtask") or "")
                by_ep.setdefault(ep, []).append(
                    (int(obj["start_frame"]), int(obj["end_frame"]), text)
                )
        frozen = {ep: tuple(sorted(spans, key=lambda s: s[0])) for ep, spans in by_ep.items()}
        return cls(spans=frozen)

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data or "frame_index" not in data:
            return data
        ep = _as_int(data["episode_index"])
        fr = _as_int(data["frame_index"])
        for t0, t1, text in self.spans.get(ep, ()):
            if t0 <= fr < t1:
                return {**data, "prompt": text}
        return data


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
""",
        label="PromptFromSubtaskSpans",
    )

    yml = ROOT / "scripts/train_from_yaml.py"
    once(
        yml,
        """    print(f"repo_id:     {getattr(cfg.data, 'repo_id', None)}")
    print(f"yaml:        {yaml_path}")
    print("==================================================")
""",
        """    print(f"repo_id:     {getattr(cfg.data, 'repo_id', None)}")
    data_cfg = cfg.data
    prompt_task = bool(getattr(data_cfg, "prompt_from_task", False))
    prompt_sub = bool(getattr(data_cfg, "prompt_from_subtask", False))
    repo_id = getattr(data_cfg, "repo_id", None)
    print(f"prompt_from_task:    {prompt_task}")
    print(f"prompt_from_subtask: {prompt_sub}")
    if prompt_sub:
        from openpi.training.data_loader import inspect_subtask_sidecar

        cov = inspect_subtask_sidecar(str(repo_id) if repo_id else None)
        if cov["sidecar"]:
            print(
                f"subtask_sidecar:     {cov['sidecar']}  "
                f"({cov['annotated_episodes']}/{cov['total_episodes']} episodes, {cov['spans']} spans)"
            )
        else:
            print("subtask_sidecar:     MISSING — 仍用 tasks.jsonl 全局 task")
    print(f"yaml:        {yaml_path}")
    print("==================================================")
""",
        label="train_from_yaml sidecar print",
    )

    log = ROOT / "src/openpi/training/train_log.py"
    once(
        log,
        """    src = resolve_source_yaml(source_yaml)
    payload = {
        "_meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config_file": str(src.resolve()) if src else None,
            "config_basename": src.name if src else None,
            "note": "Resolved effective TrainConfig after YAML overrides on the named preset.",
        },
        "params": to_jsonable(config),
    }
""",
        """    src = resolve_source_yaml(source_yaml)
    data = getattr(config, "data", None)
    prompt_meta = {
        "prompt_from_task": bool(getattr(data, "prompt_from_task", False)) if data is not None else None,
        "prompt_from_subtask": bool(getattr(data, "prompt_from_subtask", False)) if data is not None else None,
        "repo_id": getattr(data, "repo_id", None) if data is not None else None,
    }
    try:
        from openpi.training.data_loader import inspect_subtask_sidecar

        prompt_meta["subtask_sidecar"] = inspect_subtask_sidecar(prompt_meta.get("repo_id"))
    except Exception:
        prompt_meta["subtask_sidecar"] = None
    payload = {
        "_meta": {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "config_file": str(src.resolve()) if src else None,
            "config_basename": src.name if src else None,
            "note": "Resolved effective TrainConfig after YAML overrides on the named preset.",
            "prompt": prompt_meta,
        },
        "params": to_jsonable(config),
    }
""",
        label="train_log prompt meta",
    )
    print("[OK] openpi-jiangsuan patched")


if __name__ == "__main__":
    main()
