#!/usr/bin/env python3
"""Sanity-check TongBot π0.5 RGB/DEPTH/FT modality presets (no GPU, no dataset)."""

from __future__ import annotations

import openpi.policies.aloha_policy as aloha_policy
import openpi.training.config as config
import openpi.transforms as transforms

EXPECTED = {
    "pi05_aloha_rgb": {"depth": "none", "ft": False, "keys": (), "tags": {}},
    "pi05_aloha_rgb_depth_head": {
        "depth": "head",
        "ft": False,
        "keys": ("observation.depths.cam_mid",),
        "tags": {"cam_mid_depth": "DEPTH HEAD"},
    },
    "pi05_aloha_rgb_depth_left": {
        "depth": "left",
        "ft": False,
        "keys": ("observation.depths.cam_left",),
        "tags": {"cam_left_depth": "DEPTH WRIST LEFT"},
    },
    "pi05_aloha_rgb_depth_right": {
        "depth": "right",
        "ft": False,
        "keys": ("observation.depths.cam_right",),
        "tags": {"cam_right_depth": "DEPTH WRIST RIGHT"},
    },
    "pi05_aloha_rgb_ft": {
        "depth": "none",
        "ft": True,
        "keys": (),
        "tags": {"force6d.left": "FT LEFT", "force6d.right": "FT RIGHT"},
    },
    "pi05_aloha_rgb_depth_head_ft": {
        "depth": "head",
        "ft": True,
        "keys": ("observation.depths.cam_mid",),
        "tags": {
            "cam_mid_depth": "DEPTH HEAD",
            "force6d.left": "FT LEFT",
            "force6d.right": "FT RIGHT",
        },
    },
    "pi05_aloha_rgb_depth_left_ft": {
        "depth": "left",
        "ft": True,
        "keys": ("observation.depths.cam_left",),
        "tags": {
            "cam_left_depth": "DEPTH WRIST LEFT",
            "force6d.left": "FT LEFT",
            "force6d.right": "FT RIGHT",
        },
    },
    "pi05_aloha_rgb_depth_right_ft": {
        "depth": "right",
        "ft": True,
        "keys": ("observation.depths.cam_right",),
        "tags": {
            "cam_right_depth": "DEPTH WRIST RIGHT",
            "force6d.left": "FT LEFT",
            "force6d.right": "FT RIGHT",
        },
    },
}


def _repack_structure(cfg: config.TrainConfig) -> dict:
    for t in cfg.data.repack_transforms.inputs:
        if isinstance(t, transforms.RepackTransform):
            return t.structure
    raise AssertionError(f"{cfg.name}: missing RepackTransform")


def _sidecar(cfg: config.TrainConfig) -> aloha_policy.LoadSidecarDepthPNGs | None:
    found = [t for t in cfg.data.repack_transforms.inputs if isinstance(t, aloha_policy.LoadSidecarDepthPNGs)]
    if len(found) > 1:
        raise AssertionError(f"{cfg.name}: multiple LoadSidecarDepthPNGs")
    return found[0] if found else None


def main() -> None:
    for name, spec in EXPECTED.items():
        cfg = config.get_config(name)
        assert cfg.model.pi05
        assert cfg.model.use_depth_encoder is False
        assert bool(cfg.model.use_force6d_encoder) is spec["ft"], name
        structure = _repack_structure(cfg)
        assert "images" in structure, name
        assert set(structure["images"]) == {
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        }
        sidecar = _sidecar(cfg)
        if spec["depth"] == "none":
            assert sidecar is None, name
            assert "depths" not in structure, name
        else:
            assert sidecar is not None, name
            assert sidecar.depth_keys == spec["keys"], (name, sidecar.depth_keys)
            assert set(structure["depths"].values()) == set(spec["keys"]), name
        if spec["ft"]:
            assert structure.get("force6d") == {
                "left": "observation.force6d.left",
                "right": "observation.force6d.right",
            }, name
        else:
            assert "force6d" not in structure, name
        assert dict(cfg.data.modality_prompt_tags) == spec["tags"], (name, cfg.data.modality_prompt_tags)
        assert dict(cfg.model.modality_prompt_tags) == spec["tags"], (name, cfg.model.modality_prompt_tags)
        assert cfg.data.append_modality_prompt is False, name
        assert cfg.model.use_modality_prompt_tokens is False, name
        print(f"ok  {name:32s}  depth={spec['depth']:5s}  ft={spec['ft']}  tags={spec['tags']}")

    import pathlib
    import yaml

    yaml_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    yaml_expected = {
        "train_pi05_rgb.yaml": (False, {}),
        "train_pi05_rgb_depth_head.yaml": (True, {"cam_mid_depth": "DEPTH HEAD"}),
        "train_pi05_rgb_depth_left.yaml": (True, {"cam_left_depth": "DEPTH WRIST LEFT"}),
        "train_pi05_rgb_depth_right.yaml": (True, {"cam_right_depth": "DEPTH WRIST RIGHT"}),
        "train_pi05_rgb_ft.yaml": (True, {"force6d.left": "FT LEFT", "force6d.right": "FT RIGHT"}),
        "train_pi05_rgb_depth_head_ft.yaml": (
            True,
            {"cam_mid_depth": "DEPTH HEAD", "force6d.left": "FT LEFT", "force6d.right": "FT RIGHT"},
        ),
        "train_pi05_rgb_depth_left_ft.yaml": (
            True,
            {
                "cam_left_depth": "DEPTH WRIST LEFT",
                "force6d.left": "FT LEFT",
                "force6d.right": "FT RIGHT",
            },
        ),
        "train_pi05_rgb_depth_right_ft.yaml": (
            True,
            {
                "cam_right_depth": "DEPTH WRIST RIGHT",
                "force6d.left": "FT LEFT",
                "force6d.right": "FT RIGHT",
            },
        ),
    }
    for fname, (flag, tags) in yaml_expected.items():
        raw = yaml.safe_load((yaml_dir / fname).read_text(encoding="utf-8"))
        data = raw["data"]
        assert data["append_modality_prompt"] is flag, fname
        assert dict(data.get("modality_prompt_tags") or {}) == tags, fname
        print(f"ok  yaml {fname}")

    print("all 8 modality presets ok")


if __name__ == "__main__":
    main()
