#!/usr/bin/env python3
"""Sanity-check TongBot π0.5 RGB/DEPTH/FT modality presets (no GPU, no dataset)."""

from __future__ import annotations

import openpi.policies.aloha_policy as aloha_policy
import openpi.training.config as config
import openpi.transforms as transforms

EXPECTED = {
    "pi05_aloha_rgb": {"depth": "none", "ft": False, "keys": (), "tags": ()},
    "pi05_aloha_rgb_depth_head": {
        "depth": "head",
        "ft": False,
        "keys": ("observation.depths.cam_mid",),
        "tags": ("DEPTH HEAD",),
    },
    "pi05_aloha_rgb_depth_wrist": {
        "depth": "wrist",
        "ft": False,
        "keys": ("observation.depths.cam_left", "observation.depths.cam_right"),
        "tags": ("DEPTH WRIST LEFT", "DEPTH WRIST RIGHT"),
    },
    "pi05_aloha_rgb_ft": {
        "depth": "none",
        "ft": True,
        "keys": (),
        "tags": ("FT LEFT", "FT RIGHT"),
    },
    "pi05_aloha_rgb_depth_head_ft": {
        "depth": "head",
        "ft": True,
        "keys": ("observation.depths.cam_mid",),
        "tags": ("DEPTH HEAD", "FT LEFT", "FT RIGHT"),
    },
    "pi05_aloha_rgb_depth_wrist_ft": {
        "depth": "wrist",
        "ft": True,
        "keys": ("observation.depths.cam_left", "observation.depths.cam_right"),
        "tags": ("DEPTH WRIST LEFT", "DEPTH WRIST RIGHT", "FT LEFT", "FT RIGHT"),
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
        assert tuple(cfg.data.modality_prompt_tags) == spec["tags"], (
            name,
            cfg.data.modality_prompt_tags,
        )
        assert cfg.data.append_modality_prompt is False, name
        print(f"ok  {name:32s}  depth={spec['depth']:5s}  ft={spec['ft']}  tags={spec['tags']}")
    tagged = aloha_policy.AppendModalityPrompt(tags=("DEPTH WRIST LEFT", "DEPTH WRIST RIGHT"))(
        {"prompt": "pick banana"}
    )
    assert tagged["prompt"] == "pick banana DEPTH WRIST LEFT DEPTH WRIST RIGHT"
    skipped = aloha_policy.AppendModalityPrompt(tags=())({"prompt": "pick banana"})
    assert skipped["prompt"] == "pick banana"

    import pathlib
    import yaml

    yaml_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    yaml_expected = {
        "train_pi05_rgb.yaml": (False, []),
        "train_pi05_rgb_depth_head.yaml": (True, ["DEPTH HEAD"]),
        "train_pi05_rgb_depth_wrist.yaml": (True, ["DEPTH WRIST LEFT", "DEPTH WRIST RIGHT"]),
        "train_pi05_rgb_ft.yaml": (True, ["FT LEFT", "FT RIGHT"]),
        "train_pi05_rgb_depth_head_ft.yaml": (True, ["DEPTH HEAD", "FT LEFT", "FT RIGHT"]),
        "train_pi05_rgb_depth_wrist_ft.yaml": (
            True,
            ["DEPTH WRIST LEFT", "DEPTH WRIST RIGHT", "FT LEFT", "FT RIGHT"],
        ),
    }
    for fname, (flag, tags) in yaml_expected.items():
        raw = yaml.safe_load((yaml_dir / fname).read_text(encoding="utf-8"))
        data = raw["data"]
        assert data["append_modality_prompt"] is flag, fname
        assert list(data.get("modality_prompt_tags") or []) == tags, fname
        print(f"ok  yaml {fname}")

    print("all 6 modality presets ok")


if __name__ == "__main__":
    main()
