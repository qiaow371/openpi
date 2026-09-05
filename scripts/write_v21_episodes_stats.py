#!/usr/bin/env python3
"""Write LeRobot v2.1 meta/episodes_stats.jsonl from per-episode parquet.

Video channel stats are filled with [0, 1] placeholders; OpenPI recomputes
state/action norm_stats separately. Numeric keys are computed from parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


VIDEO_KEYS = (
    "observation.images.front",
    "observation.images.left",
    "observation.images.right",
)
NUMERIC_KEYS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def _stats(arr: np.ndarray) -> dict:
    x = np.asarray(arr)
    if x.ndim == 1:
        x = x[:, None]
    q = np.quantile(x, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": np.min(x, axis=0).reshape(-1).tolist(),
        "max": np.max(x, axis=0).reshape(-1).tolist(),
        "mean": np.mean(x, axis=0).reshape(-1).tolist(),
        "std": np.std(x, axis=0).reshape(-1).tolist(),
        "count": [int(x.shape[0])],
        "q01": q[0].reshape(-1).tolist(),
        "q10": q[1].reshape(-1).tolist(),
        "q50": q[2].reshape(-1).tolist(),
        "q90": q[3].reshape(-1).tolist(),
        "q99": q[4].reshape(-1).tolist(),
    }


def _ch(v: float) -> list:
    # LeRobot aggregate_stats requires image/video stats shape (3, 1, 1)
    return [[[float(v)]], [[float(v)]], [[float(v)]]]


def _video_placeholder(n: int) -> dict:
    return {
        "min": _ch(0.0),
        "max": _ch(1.0),
        "mean": _ch(0.5),
        "std": _ch(0.25),
        "count": [int(n)],
        "q01": _ch(0.0),
        "q10": _ch(0.1),
        "q50": _ch(0.5),
        "q90": _ch(0.9),
        "q99": _ch(1.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--template-info", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if args.template_info:
        tmpl = json.loads(Path(args.template_info).read_text(encoding="utf-8"))
        fps = int(info.get("fps", 10))
        feats = tmpl.get("features") or {}
        for spec in feats.values():
            if isinstance(spec, dict) and spec.get("fps") is not None:
                spec["fps"] = fps
            if isinstance(spec, dict) and isinstance(spec.get("info"), dict):
                if "video.fps" in spec["info"]:
                    spec["info"]["video.fps"] = fps
                    spec["info"]["video.codec"] = "h264"
        info["features"] = feats
        info["total_videos"] = int(info["total_episodes"]) * 3
        info["total_chunks"] = 1
        info_path.write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")

    out = root / "meta" / "episodes_stats.jsonl"
    if out.exists():
        out.unlink()
    n_eps = int(info["total_episodes"])
    for ei in range(n_eps):
        pq_path = root / "data" / "chunk-000" / f"episode_{ei:06d}.parquet"
        table = pq.read_table(pq_path)
        pdf = table.to_pandas()
        n = len(pdf)
        stats: dict = {k: _video_placeholder(n) for k in VIDEO_KEYS}
        for key in NUMERIC_KEYS:
            if key not in pdf.columns:
                continue
            stats[key] = _stats(np.stack(pdf[key].to_numpy()) if key in ("observation.state", "action") else pdf[key].to_numpy())
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"episode_index": ei, "stats": stats}, ensure_ascii=False) + "\n")
        if (ei + 1) % 20 == 0 or ei == 0:
            print(f"stats episode {ei}/{n_eps - 1}", flush=True)
    print(f"wrote {out} episodes={n_eps}")


if __name__ == "__main__":
    main()
