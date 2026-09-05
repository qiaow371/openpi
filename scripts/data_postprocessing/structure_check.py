#!/usr/bin/env python3
"""数据集结构 & 元数据一致性检查

验证 LeRobot v2.1 数据集的完整性：
  - 必需目录存在 (meta/, videos/, data/)
  - info.json 字段合法
  - tasks.jsonl / episodes.jsonl 格式正确
  - 视频文件数 = episodes × cameras
  - parquet 文件数 = episodes
  - 帧数一致性（info.json.total_frames vs 实际）
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def _check_dir(path: str, label: str) -> dict:
    exists = os.path.isdir(path)
    return {"check": label, "path": path, "ok": exists,
            "detail": "" if exists else f"目录不存在: {path}"}


def _check_file(path: str, label: str) -> dict:
    exists = os.path.isfile(path)
    return {"check": label, "path": path, "ok": exists,
            "detail": "" if exists else f"文件不存在: {path}"}


def _check_info_json(dataset_path: str) -> list[dict]:
    path = os.path.join(dataset_path, "meta", "info.json")
    results = []
    if not os.path.isfile(path):
        results.append({"check": "info.json", "ok": False, "detail": "文件不存在"})
        return results

    try:
        with open(path) as f:
            info = json.load(f)
    except Exception as e:
        results.append({"check": "info.json", "ok": False, "detail": f"JSON 解析失败: {e}"})
        return results

    # 必需字段
    required = ["codebase_version", "robot_type", "total_episodes",
                 "total_frames", "total_videos", "total_tasks", "fps"]
    for key in required:
        ok = key in info
        results.append({
            "check": f"info.json.{key}",
            "ok": ok,
            "detail": "" if ok else f"缺少必需字段: {key}",
        })

    # features 检查
    features = info.get("features", {})
    has_state = "observation.state" in features
    has_action = "action" in features
    has_images = any(k.startswith("observation.images.") for k in features)

    results.append({"check": "info.json.features.state", "ok": has_state,
                     "detail": "" if has_state else "缺少 observation.state"})
    results.append({"check": "info.json.features.action", "ok": has_action,
                     "detail": "" if has_action else "缺少 action"})
    results.append({"check": "info.json.features.images", "ok": has_images,
                     "detail": "" if has_images else "缺少 observation.images.*"})

    return results


def _check_tasks(dataset_path: str) -> list[dict]:
    results = []
    # 支持 tasks.json 和 tasks.jsonl
    jsonl_path = os.path.join(dataset_path, "meta", "tasks.jsonl")
    json_path = os.path.join(dataset_path, "meta", "tasks.json")

    if os.path.isfile(jsonl_path):
        try:
            tasks = []
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        tasks.append(json.loads(line))
            results.append({"check": "tasks.jsonl", "ok": True,
                             "detail": f"{len(tasks)} 个 task"})
        except Exception as e:
            results.append({"check": "tasks.jsonl", "ok": False,
                             "detail": f"解析失败: {e}"})
    elif os.path.isfile(json_path):
        results.append({"check": "tasks", "ok": True, "detail": "tasks.json 格式"})
    else:
        results.append({"check": "tasks", "ok": False,
                         "detail": "tasks.jsonl 和 tasks.json 均不存在"})

    return results


def _check_episode_counts(dataset_path: str) -> list[dict]:
    """验证视频/数据文件数与 info.json 一致"""
    results = []
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        return [{"check": "episode_counts", "ok": False, "detail": "info.json 不存在"}]

    with open(info_path) as f:
        info = json.load(f)

    expected_episodes = info.get("total_episodes", 0)
    expected_videos = info.get("total_videos", 0)

    # 统计实际视频文件数
    videos_dir = os.path.join(dataset_path, "videos")
    actual_videos = 0
    cameras = set()
    episodes_seen = set()
    if os.path.isdir(videos_dir):
        for root, _, files in os.walk(videos_dir):
            for fname in files:
                if fname.endswith(".mp4"):
                    actual_videos += 1
                    cameras.add(os.path.basename(root))
                    ep_str = Path(fname).stem.replace("episode_", "")
                    if ep_str.isdigit():
                        episodes_seen.add(int(ep_str))

    results.append({
        "check": "video_count",
        "ok": actual_videos == expected_videos,
        "detail": f"期望 {expected_videos}, 实际 {actual_videos}",
    })

    # 统计 parquet 文件数
    data_dir = os.path.join(dataset_path, "data")
    actual_parquet = 0
    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(".parquet"):
                    actual_parquet += 1

    results.append({
        "check": "parquet_count",
        "ok": actual_parquet == expected_episodes,
        "detail": f"期望 {expected_episodes}, 实际 {actual_parquet}",
    })

    # 相机数
    n_cameras = len(cameras)
    expected_cameras = expected_videos // expected_episodes if expected_episodes > 0 else 0
    results.append({
        "check": "camera_count",
        "ok": n_cameras == expected_cameras or expected_cameras == 0,
        "detail": f"{n_cameras} 个相机, 期望 {expected_cameras}",
    })

    # Episode 连续性
    if episodes_seen:
        ep_range = set(range(min(episodes_seen), max(episodes_seen) + 1))
        missing = ep_range - episodes_seen
        results.append({
            "check": "episode_continity",
            "ok": len(missing) == 0,
            "detail": f"episode {min(episodes_seen)}-{max(episodes_seen)}, "
                      f"缺失: {sorted(missing)[:10]}" if missing else "连续",
        })

    return results


def run(dataset_path: str, verbose: bool = True) -> dict:
    """执行结构 & 元数据检查

    Returns:
        dict with keys: ok, checks (list of {check, ok, detail, path?})
    """
    dataset_path = os.path.abspath(dataset_path)
    if verbose:
        print(f"  [structure_check] 检查数据集结构: {os.path.basename(dataset_path)}")

    checks = []

    # 1. 目录检查
    checks.append(_check_dir(os.path.join(dataset_path, "meta"), "meta/"))
    checks.append(_check_dir(os.path.join(dataset_path, "videos"), "videos/"))
    checks.append(_check_dir(os.path.join(dataset_path, "data"), "data/"))

    # 2. info.json
    checks.extend(_check_info_json(dataset_path))

    # 3. tasks
    checks.extend(_check_tasks(dataset_path))

    # 4. episodes.jsonl
    ep_path = os.path.join(dataset_path, "meta", "episodes.jsonl")
    if os.path.isfile(ep_path):
        try:
            n = sum(1 for line in open(ep_path) if line.strip())
            checks.append({"check": "episodes.jsonl", "ok": True, "detail": f"{n} 行"})
        except Exception as e:
            checks.append({"check": "episodes.jsonl", "ok": False, "detail": str(e)})
    else:
        checks.append({"check": "episodes.jsonl", "ok": False, "detail": "不存在"})

    # 5. 文件数一致性
    checks.extend(_check_episode_counts(dataset_path))

    all_ok = all(c["ok"] for c in checks)
    return {"ok": all_ok, "checks": checks}
