#!/usr/bin/env python3
"""Convert LeRobot v3 (sharded) to v2.1 and resample action/video to a target fps.

15→10: keep frames at src_idx = round(k * src_fps / dst_fps) so each training
step is 100ms. LeRobot 0.1.0 requires delta_timestamps to land on 1/fps frames,
so we must rewrite the dataset instead of only changing info.json.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CAMS = (
    "observation.images.front",
    "observation.images.left",
    "observation.images.right",
)


def subsample_indices(n_src: int, src_fps: float, dst_fps: float) -> np.ndarray:
    n_dst = max(1, int(round(n_src * dst_fps / src_fps)))
    idx = np.round(np.arange(n_dst) * src_fps / dst_fps).astype(np.int64)
    return np.clip(idx, 0, n_src - 1)


def ffmpeg_resample_clip(src: Path, dst: Path, t0: float, t1: float, fps: int) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1.0 / fps, float(t1) - float(t0))
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{float(t0):.6f}",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-vf",
        f"fps={int(fps)}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(dst),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(probe.stdout.strip() or "0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--src-fps", type=float, default=15.0)
    parser.add_argument("--dst-fps", type=float, default=10.0)
    args = parser.parse_args()
    src = args.src.expanduser().resolve()
    dst = args.dst.expanduser().resolve()
    src_fps = float(args.src_fps)
    dst_fps = float(args.dst_fps)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    ep_table = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet")
    ep = ep_table.to_pydict()
    data = pq.read_table(src / "data/chunk-000/file-000.parquet").to_pandas()
    n_eps = int(info["total_episodes"])
    task = "With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box."
    if ep.get("tasks") and ep["tasks"][0]:
        t0 = ep["tasks"][0]
        task = t0[0] if isinstance(t0, (list, tuple)) else str(t0)

    out_rows = 0
    episodes_meta = []
    chunk = 0
    data_dir = dst / f"data/chunk-{chunk:03d}"
    data_dir.mkdir(parents=True)

    for i in range(n_eps):
        ei = int(ep["episode_index"][i])
        a = int(ep["dataset_from_index"][i])
        b = int(ep["dataset_to_index"][i])
        part = data.iloc[a:b].reset_index(drop=True)
        n_src = len(part)
        idx = subsample_indices(n_src, src_fps, dst_fps)
        sub = part.iloc[idx].copy().reset_index(drop=True)
        n_out = len(sub)
        sub["episode_index"] = ei
        sub["frame_index"] = np.arange(n_out, dtype=np.int64)
        sub["timestamp"] = np.arange(n_out, dtype=np.float64) / dst_fps
        sub["index"] = np.arange(out_rows, out_rows + n_out, dtype=np.int64)
        if "task_index" not in sub.columns:
            sub["task_index"] = 0

        pq_path = data_dir / f"episode_{ei:06d}.parquet"
        sub.to_parquet(pq_path, index=False)

        for cam in CAMS:
            v_chunk = int(ep[f"videos/{cam}/chunk_index"][i])
            v_file = int(ep[f"videos/{cam}/file_index"][i])
            t0 = float(ep[f"videos/{cam}/from_timestamp"][i])
            t1 = float(ep[f"videos/{cam}/to_timestamp"][i])
            shard = src / "videos" / cam / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
            out_mp4 = dst / "videos" / f"chunk-{chunk:03d}" / cam / f"episode_{ei:06d}.mp4"
            n_vid = ffmpeg_resample_clip(shard, out_mp4, t0, t1, int(dst_fps))
            if n_vid != n_out and n_vid > 0:
                n_keep = min(n_out, n_vid)
                sub = sub.iloc[:n_keep].copy()
                sub["frame_index"] = np.arange(n_keep, dtype=np.int64)
                sub["timestamp"] = np.arange(n_keep, dtype=np.float64) / dst_fps
                sub.to_parquet(pq_path, index=False)
                n_out = n_keep
        out_rows += n_out
        episodes_meta.append({"episode_index": ei, "tasks": [task], "length": int(n_out)})
        if (i + 1) % 10 == 0 or i == 0:
            print(f"episode {ei}/{n_eps - 1} frames {n_src}->{n_out}", flush=True)

    features = info.get("features") or {}
    for key, spec in features.items():
        if isinstance(spec, dict) and spec.get("fps") is not None:
            spec["fps"] = int(dst_fps)
        if isinstance(spec, dict) and isinstance(spec.get("info"), dict) and "video.fps" in spec["info"]:
            spec["info"]["video.fps"] = int(dst_fps)
            spec["info"]["video.codec"] = "h264"
    out_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", "tong_dual_arm"),
        "total_episodes": n_eps,
        "total_frames": int(out_rows),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": int(dst_fps),
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "source_fps": src_fps,
        "resampled_from": str(src),
    }
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(out_info, indent=4) + "\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"done dst={dst} episodes={n_eps} frames={out_rows} fps={dst_fps}")


if __name__ == "__main__":
    main()
