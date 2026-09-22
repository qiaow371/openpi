#!/usr/bin/env python3
"""Sanity-check TongBot π0.5 RGB/DEPTH/FT modality presets (no GPU, no dataset)."""

from __future__ import annotations

import openpi.policies.aloha_policy as aloha_policy
import openpi.training.config as config
import openpi.transforms as transforms

EXPECTED = {
    "pi05_aloha_rgb": {"depth": "none", "ft": False, "keys": ()},
    "pi05_aloha_rgb_depth_head": {
        "depth": "head",
        "ft": False,
        "keys": ("observation.depths.cam_mid",),
    },
    "pi05_aloha_rgb_depth_wrist": {
        "depth": "wrist",
        "ft": False,
        "keys": ("observation.depths.cam_left", "observation.depths.cam_right"),
    },
    "pi05_aloha_rgb_ft": {"depth": "none", "ft": True, "keys": ()},
    "pi05_aloha_rgb_depth_head_ft": {
        "depth": "head",
        "ft": True,
        "keys": ("observation.depths.cam_mid",),
    },
    "pi05_aloha_rgb_depth_wrist_ft": {
        "depth": "wrist",
        "ft": True,
        "keys": ("observation.depths.cam_left", "observation.depths.cam_right"),
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
        print(f"ok  {name:32s}  depth={spec['depth']:5s}  ft={spec['ft']}")
    print("all 6 modality presets ok")


if __name__ == "__main__":
    main()
