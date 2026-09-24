#!/usr/bin/env python3
"""Sanity-check TongBot π0.5 RGB/DEPTH/FT modality presets (no GPU, no dataset)."""

from __future__ import annotations

import dataclasses

import numpy as np

import openpi.policies.aloha_policy as aloha_policy
import openpi.training.config as config
import openpi.transforms as transforms

EXPECTED = {
    "pi05_aloha_rgb": {"depth": "none", "ft": False, "keys": (), "tags": {}},
    "pi05_aloha_rgb_depth_head": {
        "depth": "head",
        "ft": False,
        "keys": ("observation.depths.cam_mid",),
        "tags": {"cam_mid_depth": "DEPTH HEAD: "},
    },
    "pi05_aloha_rgb_depth_left": {
        "depth": "left",
        "ft": False,
        "keys": ("observation.depths.cam_left",),
        "tags": {"cam_left_depth": "DEPTH WRIST LEFT: "},
    },
    "pi05_aloha_rgb_depth_right": {
        "depth": "right",
        "ft": False,
        "keys": ("observation.depths.cam_right",),
        "tags": {"cam_right_depth": "DEPTH WRIST RIGHT: "},
    },
    "pi05_aloha_rgb_ft": {
        "depth": "none",
        "ft": True,
        "keys": (),
        "tags": {"force6d.left": "FORCE TORQUE LEFT: ", "force6d.right": "FORCE TORQUE RIGHT: "},
    },
    "pi05_aloha_rgb_depth_head_ft": {
        "depth": "head",
        "ft": True,
        "keys": ("observation.depths.cam_mid",),
        "tags": {
            "cam_mid_depth": "DEPTH HEAD: ",
            "force6d.left": "FORCE TORQUE LEFT: ",
            "force6d.right": "FORCE TORQUE RIGHT: ",
        },
    },
    "pi05_aloha_rgb_depth_left_ft": {
        "depth": "left",
        "ft": True,
        "keys": ("observation.depths.cam_left",),
        "tags": {
            "cam_left_depth": "DEPTH WRIST LEFT: ",
            "force6d.left": "FORCE TORQUE LEFT: ",
            "force6d.right": "FORCE TORQUE RIGHT: ",
        },
    },
    "pi05_aloha_rgb_depth_right_ft": {
        "depth": "right",
        "ft": True,
        "keys": ("observation.depths.cam_right",),
        "tags": {
            "cam_right_depth": "DEPTH WRIST RIGHT: ",
            "force6d.left": "FORCE TORQUE LEFT: ",
            "force6d.right": "FORCE TORQUE RIGHT: ",
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
        assert cfg.data.visual_norm == "siglip", name
        assert cfg.data.vector_norm == "quantile", name
        print(f"ok  {name:32s}  depth={spec['depth']:5s}  ft={spec['ft']}  tags={spec['tags']}")

    import pathlib
    import yaml

    yaml_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    yaml_expected = {
        "train_pi05_rgb.yaml": (False, {}),
        "train_pi05_rgb_depth_head.yaml": (True, {"cam_mid_depth": "DEPTH HEAD: "}),
        "train_pi05_rgb_depth_left.yaml": (True, {"cam_left_depth": "DEPTH WRIST LEFT: "}),
        "train_pi05_rgb_depth_right.yaml": (True, {"cam_right_depth": "DEPTH WRIST RIGHT: "}),
        "train_pi05_rgb_ft.yaml": (True, {"force6d.left": "FORCE TORQUE LEFT: ", "force6d.right": "FORCE TORQUE RIGHT: "}),
        "train_pi05_rgb_depth_head_ft.yaml": (
            True,
            {"cam_mid_depth": "DEPTH HEAD: ", "force6d.left": "FORCE TORQUE LEFT: ", "force6d.right": "FORCE TORQUE RIGHT: "},
        ),
        "train_pi05_rgb_depth_left_ft.yaml": (
            True,
            {
                "cam_left_depth": "DEPTH WRIST LEFT: ",
                "force6d.left": "FORCE TORQUE LEFT: ",
                "force6d.right": "FORCE TORQUE RIGHT: ",
            },
        ),
        "train_pi05_rgb_depth_right_ft.yaml": (
            True,
            {
                "cam_right_depth": "DEPTH WRIST RIGHT: ",
                "force6d.left": "FORCE TORQUE LEFT: ",
                "force6d.right": "FORCE TORQUE RIGHT: ",
            },
        ),
    }
    for fname, (flag, tags) in yaml_expected.items():
        raw = yaml.safe_load((yaml_dir / fname).read_text(encoding="utf-8"))
        data = raw["data"]
        assert data["append_modality_prompt"] is flag, fname
        assert dict(data.get("modality_prompt_tags") or {}) == tags, fname
        assert data["visual_norm"] == "siglip", fname
        assert data["vector_norm"] == "quantile", fname
        assert float(data["max_depth_mm"]) == 4000.0, fname
        print(f"ok  yaml {fname}")

    rgb = config.get_config("pi05_aloha_rgb")
    created = rgb.data.create(pathlib.Path("/tmp"), rgb.model)
    assert created.visual_norm == "siglip"
    assert created.vector_norm == "quantile"
    assert created.use_quantile_norm is True
    created_z = dataclasses.replace(rgb.data, vector_norm="zscore").create(pathlib.Path("/tmp"), rgb.model)
    assert created_z.vector_norm == "zscore"
    assert created_z.use_quantile_norm is False
    print("ok  vector_norm quantile/zscore → use_quantile_norm")

    _check_depth_mm_from_stats()
    _smoke_all_option_transforms()
    print("all 8 modality presets ok")


def _write_uint16_png(path, value: int = 1200) -> None:
    import cv2
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((8, 8), value, dtype=np.uint16)
    assert cv2.imwrite(str(path), img)


def _fake_sample(root, *, use_underscore: bool = True) -> dict:
    import numpy as np

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb[..., 1] = 40
    sample = {
        "observation.images.cam_mid": rgb.copy(),
        "observation.images.cam_left": rgb.copy(),
        "observation.images.cam_right": rgb.copy(),
        "observation.state": np.zeros(16, dtype=np.float32),
        "action": np.zeros(16, dtype=np.float32),
        "observation.force6d.left": np.array([1, 0, 5, 0, 0, 0], dtype=np.float32),
        "observation.force6d.right": np.array([0, 1, 8, 0, 0, 0], dtype=np.float32),
        "episode_index": np.array(0),
        "frame_index": np.array(0),
        "prompt": "pick banana",
    }
    name = "frame_000000.png" if use_underscore else "frame-000000.png"
    for cam in ("cam_left", "cam_mid", "cam_right"):
        key = f"observation.depths.{cam}"
        _write_uint16_png(
            root / "depth" / key / "chunk-000" / "episode-000000" / name,
            1500,
        )
    return sample


def _apply_option(cfg: config.TrainConfig, sample: dict, dataset_root: str) -> dict:
    import copy

    data = copy.deepcopy(sample)
    for t in cfg.data.repack_transforms.inputs:
        if isinstance(t, aloha_policy.LoadSidecarDepthPNGs):
            t = dataclasses.replace(t, dataset_root=dataset_root)
        data = t(data)
    extra = []
    if any(isinstance(t, aloha_policy.LoadSidecarDepthPNGs) for t in cfg.data.repack_transforms.inputs):
        extra.extend([aloha_policy.ProcessDepths(), aloha_policy.DepthsAsSiglipImages()])
    if cfg.model.use_force6d_encoder:
        extra.append(aloha_policy.ProcessForce6D())
    extra.append(aloha_policy.AlohaInputs(adapt_to_pi=False))
    for t in extra:
        data = t(data)
    return data


def _smoke_all_option_transforms() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "meta").mkdir()
        (root / "meta" / "info.json").write_text('{"chunks_size": 1000, "features": {}}', encoding="utf-8")
        sample = _fake_sample(root, use_underscore=True)
        for name, spec in EXPECTED.items():
            cfg = config.get_config(name)
            out = _apply_option(cfg, sample, str(root))
            images = out["image"]
            assert set(aloha_policy.AlohaInputs.STANDARD_IMAGE_KEYS).issubset(images), name
            depth_keys = {f"{cam}_depth" for cam in ("cam_left", "cam_mid", "cam_right")}
            present = set(images) & depth_keys
            if spec["depth"] == "none":
                assert not present, (name, present)
            elif spec["depth"] == "head":
                assert present == {"cam_mid_depth"}, (name, present)
            elif spec["depth"] == "left":
                assert present == {"cam_left_depth"}, (name, present)
            else:
                assert present == {"cam_right_depth"}, (name, present)
            if spec["ft"]:
                assert "force6d" in out and set(out["force6d"]) == {"left", "right"}, name
                assert bool(out["force6d_mask"]["left"]) is True, name
            else:
                assert "force6d" not in out, name
            print(f"ok  transform {name}")

        import shutil

        shutil.rmtree(root / "depth")
        hyphen = _fake_sample(root, use_underscore=False)
        out = _apply_option(config.get_config("pi05_aloha_rgb_depth_head"), hyphen, str(root))
        assert "cam_mid_depth" in out["image"]
        print("ok  transform depth filename hyphen")

        nan_sample = _fake_sample(root, use_underscore=False)
        nan_sample["observation.force6d.left"] = np.array([np.nan] * 6, dtype=np.float32)
        out = _apply_option(config.get_config("pi05_aloha_rgb_ft"), nan_sample, str(root))
        assert bool(out["force6d_mask"]["left"]) is False
        assert bool(out["force6d_mask"]["right"]) is True
        print("ok  transform ft nan mask")


def _check_depth_mm_from_stats() -> None:
    import tempfile
    from pathlib import Path

    from openpi.shared import normalize as normalize

    mm = np.full((4, 4), 1500, dtype=np.uint16)
    fallback = aloha_policy.DepthsAsSiglipImages(max_depth_mm=4000.0)
    out = fallback({"depths": {"cam_mid": mm}})
    fallback_u8 = int(out["images"]["cam_mid_depth"][0, 0, 0])
    assert fallback_u8 == int(round(1500 / 4000 * 255)), fallback_u8

    fitted = aloha_policy.DepthsAsSiglipImages(
        max_depth_mm=4000.0,
        mm_q01={"cam_mid": 500.0},
        mm_q99={"cam_mid": 2500.0},
    )
    out = fitted({"depths": {"cam_mid": mm}})
    fitted_u8 = int(out["images"]["cam_mid_depth"][0, 0, 0])
    expect = int(round((1500 - 500) / (2500 - 500) * 255))
    assert fitted_u8 == expect, (fitted_u8, expect)
    assert fitted_u8 != fallback_u8
    print("ok  depth mm q01/q99 mapping vs 4000 fallback")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        normalize.save(
            root,
            {
                "depth_mm/cam_mid": normalize.NormStats(
                    mean=np.array([1000.0]),
                    std=np.array([200.0]),
                    q01=np.array([500.0]),
                    q99=np.array([2500.0]),
                )
            },
        )
        cfg = config.get_config("pi05_aloha_rgb_depth_head")
        data = dataclasses.replace(
            cfg.data,
            repo_id=str(root),
            assets=dataclasses.replace(cfg.data.assets, assets_dir=str(root), asset_id="."),
        )
        created = data.create(root, cfg.model)
        depth_ts = [
            t for t in created.data_transforms.inputs if isinstance(t, aloha_policy.DepthsAsSiglipImages)
        ]
        assert len(depth_ts) == 1, depth_ts
        assert depth_ts[0].mm_q01.get("cam_mid") == 500.0, depth_ts[0].mm_q01
        assert depth_ts[0].mm_q99.get("cam_mid") == 2500.0, depth_ts[0].mm_q99
        kept = transforms.filter_vector_norm_stats(created.norm_stats)
        assert "depth_mm" not in str(kept)
        print("ok  create() wires depth_mm stats; Normalize filter drops them")


if __name__ == "__main__":
    main()
