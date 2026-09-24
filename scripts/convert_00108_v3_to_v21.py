#!/usr/bin/env python3
"""LeRobot v3 (00108 cam_mid/left/right) → v2.1 layout, keep 15 fps. Depth via symlink."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMS = (
    "observation.images.cam_left",
    "observation.images.cam_mid",
    "observation.images.cam_right",
)


def ffmpeg_clip(src: Path, dst: Path, t0: float, t1: float, fps: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(1.0 / fps, float(t1) - float(t0))
    cmd = [
        "ffmpeg", "-y", "-ss", f"{float(t0):.6f}", "-i", str(src),
        "-t", f"{duration:.6f}", "-r", str(int(fps)), "-an",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "23",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> None:
    src = Path("/home/xk/dataset/00108_20260922_RM_01_YJ")
    dst = Path("/home/xk/dataset/00108_20260922_RM_01_YJ_v21")
    fps = 15
    if dst.exists():
        raise SystemExit(f"already exists: {dst}")
    dst.mkdir(parents=True)
    info = json.loads((src / "meta/info.json").read_text(encoding="utf-8"))
    ep = pq.read_table(src / "meta/episodes/chunk-000/file-000.parquet").to_pydict()
    data = pq.read_table(src / "data/chunk-000/file-000.parquet").to_pandas()
    n_eps = int(info["total_episodes"])
    task = "With both hands, pick up the yellow banana and the yellow lemon and put them into the black box; then take the green vegetables and place them into the blue box."
    if ep.get("tasks") and ep["tasks"][0]:
        t0 = ep["tasks"][0]
        task = t0[0] if isinstance(t0, (list, tuple)) else str(t0)

    jobs = []
    for i in range(n_eps):
        for cam in CAMS:
            v_chunk = int(ep[f"videos/{cam}/chunk_index"][i])
            v_file = int(ep[f"videos/{cam}/file_index"][i])
            t0 = float(ep[f"videos/{cam}/from_timestamp"][i])
            t1 = float(ep[f"videos/{cam}/to_timestamp"][i])
            shard = src / "videos" / cam / f"chunk-{v_chunk:03d}" / f"file-{v_file:03d}.mp4"
            ei = int(ep["episode_index"][i])
            out_mp4 = dst / "videos" / "chunk-000" / cam / f"episode_{ei:06d}.mp4"
            jobs.append((shard, out_mp4, t0, t1, fps))

    print(f"ffmpeg clips: {len(jobs)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = [pool.submit(ffmpeg_clip, *j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if n % 30 == 0 or n == len(futs):
                print(f"  video {n}/{len(futs)}", flush=True)

    data_dir = dst / "data/chunk-000"
    data_dir.mkdir(parents=True)
    episodes_meta = []
    out_rows = 0
    for i in range(n_eps):
        ei = int(ep["episode_index"][i])
        a = int(ep["dataset_from_index"][i])
        b = int(ep["dataset_to_index"][i])
        sub = data.iloc[a:b].copy().reset_index(drop=True)
        n_out = len(sub)
        sub["episode_index"] = ei
        sub["frame_index"] = np.arange(n_out, dtype=np.int64)
        sub["timestamp"] = np.arange(n_out, dtype=np.float64) / fps
        sub["index"] = np.arange(out_rows, out_rows + n_out, dtype=np.int64)
        if "task_index" not in sub.columns:
            sub["task_index"] = 0
        sub.to_parquet(data_dir / f"episode_{ei:06d}.parquet", index=False)
        out_rows += n_out
        episodes_meta.append({"episode_index": ei, "tasks": [task], "length": int(n_out)})

    features = info.get("features") or {}
    for spec in features.values():
        if isinstance(spec, dict) and isinstance(spec.get("info"), dict):
            spec["info"]["video.codec"] = "h264"
    out_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", "tong_dual_arm"),
        "total_episodes": n_eps,
        "total_frames": int(out_rows),
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{n_eps}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "source": str(src),
    }
    meta = dst / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(out_info, indent=4) + "\n", encoding="utf-8")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episodes_meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    depth = dst / "depth"
    if not depth.exists():
        depth.symlink_to(src / "depth")
    print(f"done dst={dst} episodes={n_eps} frames={out_rows}", flush=True)


if __name__ == "__main__":
    main()
