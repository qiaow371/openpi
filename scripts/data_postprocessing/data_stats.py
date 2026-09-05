#!/usr/bin/env python3
"""数据集统计摘要

生成 LeRobot v2.1 数据集的统计信息：
  - 基本信息 (版本, 机器人类型, FPS, episode 数, 帧数)
  - 相机列表
  - 每 episode 帧数分布
  - 视频文件大小统计
  - action/state 维度
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def run(dataset_path: str, verbose: bool = True) -> dict:
    """生成数据集统计摘要"""
    dataset_path = os.path.abspath(dataset_path)
    dataset_name = os.path.basename(dataset_path)

    if verbose:
        print(f"  [data_stats] 统计: {dataset_name}")

    stats = {"dataset_name": dataset_name, "dataset_path": dataset_path}

    # info.json
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        stats["error"] = "info.json not found"
        return stats

    with open(info_path) as f:
        info = json.load(f)

    stats["codebase_version"] = info.get("codebase_version", "?")
    stats["robot_type"] = info.get("robot_type", "?")
    stats["fps"] = info.get("fps", 0)
    stats["total_episodes"] = info.get("total_episodes", 0)
    stats["total_frames"] = info.get("total_frames", 0)
    stats["total_videos"] = info.get("total_videos", 0)
    stats["total_tasks"] = info.get("total_tasks", 0)

    # features
    features = info.get("features", {})
    state_shape = features.get("observation.state", {}).get("shape", [])
    action_shape = features.get("action", {}).get("shape", [])
    image_keys = [k for k in features if k.startswith("observation.images.")]

    stats["state_dim"] = state_shape[0] if state_shape else 0
    stats["action_dim"] = action_shape[0] if action_shape else 0
    stats["cameras"] = image_keys

    # tasks
    tasks = []
    tasks_jsonl = os.path.join(dataset_path, "meta", "tasks.jsonl")
    if os.path.isfile(tasks_jsonl):
        with open(tasks_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        t = json.loads(line)
                        tasks.append(t.get("task", ""))
                    except Exception:
                        pass
    stats["tasks"] = tasks

    # 视频文件大小统计
    videos_dir = os.path.join(dataset_path, "videos")
    sizes_kb = []
    if os.path.isdir(videos_dir):
        for root, _, files in os.walk(videos_dir):
            for fname in files:
                if fname.endswith(".mp4"):
                    sizes_kb.append(os.path.getsize(os.path.join(root, fname)) / 1024)

    if sizes_kb:
        stats["video_size"] = {
            "count": len(sizes_kb),
            "total_mb": round(sum(sizes_kb) / 1024, 1),
            "mean_kb": round(sum(sizes_kb) / len(sizes_kb), 1),
            "min_kb": round(min(sizes_kb), 1),
            "max_kb": round(max(sizes_kb), 1),
        }

    # 数据集总大小
    total_size = 0
    for root, _, files in os.walk(dataset_path):
        for fname in files:
            total_size += os.path.getsize(os.path.join(root, fname))
    stats["total_size_mb"] = round(total_size / (1024 * 1024), 1)

    # 每 episode 帧数（从 parquet 文件估算）
    ep_frames = []
    data_dir = os.path.join(dataset_path, "data")
    if os.path.isdir(data_dir):
        try:
            import pandas as pd
            for root, _, files in os.walk(data_dir):
                for fname in sorted(files):
                    if fname.endswith(".parquet"):
                        df = pd.read_parquet(os.path.join(root, fname))
                        ep_frames.append(len(df))
        except ImportError:
            pass  # pandas not available, skip

    if ep_frames:
        stats["frames_per_episode"] = {
            "mean": round(sum(ep_frames) / len(ep_frames), 1),
            "min": min(ep_frames),
            "max": max(ep_frames),
        }

    return stats
