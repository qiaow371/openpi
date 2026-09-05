#!/usr/bin/env python3
"""视频完整性检查 & 自动修复

扫描 LeRobot v2.1 数据集中所有 mp4 视频文件：
  - 全量解码检测损坏（尾部截断、moov atom 缺失、编码错误）
  - ffmpeg 重新编码修复（保留可用帧，加 faststart 防止再出问题）
  - 修复后验证 + JSON 报告
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class VideoReport:
    path: str
    camera: str
    episode: int
    status: str  # ok | corrupted | fixed | fix_failed
    frames_expected: int = 0
    frames_decoded: int = 0
    frames_lost: int = 0
    file_size_kb: float = 0
    error_message: str = ""
    repair_note: str = ""


# ── 核心检查 ─────────────────────────────────────────────────────────────

def _decode_video(path: str) -> tuple[int, Optional[str]]:
    """完整解码视频，返回 (帧数, 错误信息 or None)"""
    import av
    decoded = 0
    error = None
    try:
        container = av.open(path)
        for frame in container.decode(video=0):
            decoded += 1
        container.close()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    return decoded, error


def check_single_video(args_tuple: tuple) -> dict:
    """检查单个视频（可被 ProcessPoolExecutor 调用）"""
    import av
    video_path, rel_path, _ = args_tuple

    camera = os.path.basename(os.path.dirname(video_path))
    ep_str = Path(video_path).stem.replace("episode_", "")
    episode = int(ep_str) if ep_str.isdigit() else -1
    file_size = os.path.getsize(video_path) / 1024

    meta_frames = 0
    try:
        c = av.open(video_path)
        meta_frames = c.streams.video[0].frames or 0
        c.close()
    except Exception:
        pass

    decoded, error = _decode_video(video_path)
    status = "ok" if error is None else "corrupted"
    frames_lost = max(0, meta_frames - decoded) if meta_frames > 0 and error else 0

    return {
        "path": rel_path,
        "camera": camera,
        "episode": episode,
        "status": status,
        "frames_expected": meta_frames,
        "frames_decoded": decoded,
        "frames_lost": frames_lost,
        "file_size_kb": round(file_size, 1),
        "error_message": error or "",
        "repair_note": "",
    }


# ── 修复 ─────────────────────────────────────────────────────────────────

def fix_video(video_path: str, keep_backup: bool = True) -> tuple[bool, str]:
    """用 ffmpeg 重新编码修复视频"""
    bak_path = video_path + ".bak"
    tmp_path = video_path + ".tmp.mp4"

    if keep_backup:
        shutil.copy2(video_path, bak_path)

    cmd = [
        "ffmpeg", "-y",
        "-err_detect", "ignore_err",
        "-i", video_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        tmp_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0 and not os.path.exists(tmp_path):
            return False, f"ffmpeg 失败: {result.stderr[-200:]}"

        fixed_frames, fix_error = _decode_video(tmp_path)
        if fix_error or fixed_frames == 0:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            return False, f"修复后仍损坏: {fix_error or '0 帧'}"

        shutil.move(tmp_path, video_path)
        return True, f"修复成功 ({fixed_frames} 帧, x264+faststart)"

    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, "ffmpeg 超时 (>300s)"
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False, str(e)


# ── 入口 ─────────────────────────────────────────────────────────────────

def discover_videos(dataset_path: str) -> list[tuple[str, str, str]]:
    videos_dir = os.path.join(dataset_path, "videos")
    if not os.path.isdir(videos_dir):
        return []
    results = []
    for root, _, files in os.walk(videos_dir):
        for f in sorted(files):
            if f.endswith(".mp4"):
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, videos_dir)
                results.append((abs_path, rel_path, dataset_path))
    return results


def run(
    dataset_path: str,
    fix: bool = False,
    verify: bool = False,
    keep_backup: bool = True,
    workers: int = 1,
    verbose: bool = True,
) -> dict:
    """执行视频检查（+ 可选修复）

    Returns:
        dict with keys: ok, total, corrupted, fixed, fix_failed, videos, scan_sec, repair_sec
    """
    dataset_path = os.path.abspath(dataset_path)
    videos = discover_videos(dataset_path)

    if not videos:
        return {"ok": True, "total": 0, "corrupted": 0, "fixed": 0,
                "fix_failed": 0, "videos": [], "scan_sec": 0, "repair_sec": 0}

    if verbose:
        print(f"  [video_repair] 扫描 {len(videos)} 个视频 (workers={workers})...")

    t0 = time.time()
    reports = []

    if workers <= 1:
        for v in videos:
            reports.append(check_single_video(v))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(check_single_video, v): v for v in videos}
            for fut in as_completed(futs):
                reports.append(fut.result())

    scan_sec = round(time.time() - t0, 1)
    reports.sort(key=lambda r: (r["episode"], r["camera"]))

    corrupted_list = [r for r in reports if r["status"] == "corrupted"]
    fixed_count = 0
    failed_count = 0
    repair_sec = 0.0

    if fix and corrupted_list:
        if verbose:
            print(f"  [video_repair] 修复 {len(corrupted_list)} 个损坏视频...")
        t1 = time.time()
        videos_dir = os.path.join(dataset_path, "videos")
        for r in corrupted_list:
            abs_path = os.path.join(videos_dir, r["path"])
            ok, msg = fix_video(abs_path, keep_backup=keep_backup)
            if ok:
                r["status"] = "fixed"
                r["repair_note"] = msg
                decoded, _ = _decode_video(abs_path)
                r["frames_decoded"] = decoded
                r["frames_lost"] = max(0, r["frames_expected"] - decoded)
                r["error_message"] = ""
                fixed_count += 1
                if verbose:
                    print(f"    ✅ {r['path']}  {msg}")
            else:
                r["status"] = "fix_failed"
                r["error_message"] = msg
                failed_count += 1
                if verbose:
                    print(f"    ❌ {r['path']}  {msg}")
        repair_sec = round(time.time() - t1, 1)

        if verify:
            if verbose:
                print(f"  [video_repair] 验证修复结果...")
            for r in corrupted_list:
                if r["status"] != "fixed":
                    continue
                abs_path = os.path.join(videos_dir, r["path"])
                decoded, err = _decode_video(abs_path)
                if err:
                    r["status"] = "fix_failed"
                    r["error_message"] = f"验证失败: {err}"
                    failed_count += 1
                    fixed_count -= 1

    still_bad = sum(1 for r in reports if r["status"] in ("corrupted", "fix_failed"))

    return {
        "ok": still_bad == 0,
        "total": len(videos),
        "corrupted": len(corrupted_list),
        "fixed": fixed_count,
        "fix_failed": failed_count,
        "videos": reports,
        "scan_sec": scan_sec,
        "repair_sec": repair_sec,
    }
