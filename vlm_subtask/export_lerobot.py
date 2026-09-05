"""把 compose 的段级 subtask 写成 AgiBot 风格 sidecar，供 dataloader 按帧取 prompt。

对齐 AgiBot World ``task_info/task_*.json``：

    episode.task_name                          # 全局任务 ℓ
    episode.label_info.action_config[]         # 子任务 ˆℓ
        start_frame, end_frame, action_text, skill

并同时写 InternVLA 用的 ``meta/episodes_detailed_task.jsonl``
（每行一个 episode，顶层 ``action_config``）。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from vlm_subtask.compose import read_jsonl

DETAILED_NAME = "episodes_detailed_task.jsonl"
SPANS_NAME = "subtask_spans.jsonl"
TASKS_SUBTASK_NAME = "tasks_subtask.jsonl"


def _slice(row: dict) -> dict:
    return {
        "start_frame": int(row["t0"]),
        "end_frame": int(row["t1"]),
        "action_text": str(row["subtask"]),
        "skill": str(row.get("label") or ""),
    }


def build_agibot_episodes(rows: list[dict], *, task_id: str, bbox_rows: list[dict] | None = None) -> list[dict]:
    by_ep: dict[int, list[dict]] = defaultdict(list)
    task_name = ""
    for r in rows:
        by_ep[int(r["episode"])].append(r)
        if not task_name:
            task_name = str(r.get("task") or "")
    key_by_ep: dict[int, list[dict]] = defaultdict(list)
    for r in bbox_rows or []:
        ep = int(r.get("episode", -1))
        for kf in r.get("key_frames") or []:
            key_by_ep[ep].append(kf)
    episodes = []
    for ep in sorted(by_ep):
        segs = sorted(by_ep[ep], key=lambda x: int(x["t0"]))
        action_config = [_slice(s) for s in segs]
        episodes.append(
            {
                "episode_id": ep,
                "episode_index": ep,
                "task_id": task_id,
                "task_name": task_name or str(segs[0].get("task") or ""),
                "init_scene_text": "",
                "label_info": {
                    "action_config": action_config,
                    "key_frame": key_by_ep.get(ep, []),
                },
                "action_config": action_config,
            }
        )
    return episodes


def write_agibot(repo_root: Path, episodes: list[dict], *, task_id: str) -> dict:
    meta = repo_root / "meta"
    task_info = meta / "task_info"
    meta.mkdir(parents=True, exist_ok=True)
    task_info.mkdir(parents=True, exist_ok=True)

    task_json = task_info / f"task_{task_id}.json"
    task_json.write_text(json.dumps(episodes, indent=2, ensure_ascii=False) + "\n")

    detailed = meta / DETAILED_NAME
    with detailed.open("w") as f:
        for ep in episodes:
            f.write(
                json.dumps(
                    {
                        "episode_index": ep["episode_index"],
                        "episode_id": ep["episode_id"],
                        "task_name": ep["task_name"],
                        "action_config": ep["action_config"],
                        "key_frame": ep.get("label_info", {}).get("key_frame") or [],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    key_map = {
        str(ep["episode_index"]): {"dual": ep.get("label_info", {}).get("key_frame") or []}
        for ep in episodes
    }
    key_path = meta / "key_frame.json"
    key_path.write_text(json.dumps(key_map, indent=2, ensure_ascii=False) + "\n")
    n_kf = sum(len(v["dual"]) for v in key_map.values())
    info_path = meta / "info.json"
    if info_path.is_file() and n_kf:
        info_obj = json.loads(info_path.read_text())
        info_obj["key_frame"] = key_map
        info_path.write_text(json.dumps(info_obj, indent=2, ensure_ascii=False) + "\n")

    spans = []
    for ep in episodes:
        for sl in ep["action_config"]:
            spans.append(
                {
                    "episode_index": ep["episode_index"],
                    "start_frame": sl["start_frame"],
                    "end_frame": sl["end_frame"],
                    "subtask": sl["action_text"],
                    "action_text": sl["action_text"],
                    "label": sl["skill"],
                    "skill": sl["skill"],
                    "task": ep["task_name"],
                }
            )
    spans_file = meta / SPANS_NAME
    with spans_file.open("w") as f:
        for row in spans:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    uniq, seen = [], set()
    for s in spans:
        if s["action_text"] not in seen:
            seen.add(s["action_text"])
            uniq.append({"task_index": len(uniq), "task": s["action_text"], "skill": s["skill"]})
    tasks_path = meta / TASKS_SUBTASK_NAME
    with tasks_path.open("w") as f:
        for row in uniq:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "repo": str(repo_root),
        "format": "agibot_action_config",
        "episodes": len(episodes),
        "spans": len(spans),
        "unique_subtasks": len(uniq),
        "labels": dict(Counter(s["skill"] for s in spans)),
        "task_info": str(task_json),
        "episodes_detailed_task": str(detailed),
        "key_frame_file": str(key_path),
        "key_frames": n_kf,
        "spans_file": str(spans_file),
        "tasks_subtask_file": str(tasks_path),
    }
    (meta / "subtask_export_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def export_from_compose(compose_dir: Path, repo_root: Path) -> dict:
    rows = read_jsonl(compose_dir / "cot.jsonl")
    if not rows:
        rows = read_jsonl(compose_dir / "vlm_train.jsonl") + read_jsonl(compose_dir / "vlm_val.jsonl")
    if not rows:
        raise SystemExit(f"{compose_dir} 没有 cot.jsonl / vlm_*.jsonl，先跑 compose")
    info = repo_root / "meta" / "info.json"
    if not info.is_file():
        raise SystemExit(f"不是 LeRobot 根目录: {repo_root}")
    task_id = str(rows[0].get("dataset_id") or "task")
    bbox_rows = read_jsonl(compose_dir / "bbox.jsonl") if (compose_dir / "bbox.jsonl").is_file() else []
    episodes = build_agibot_episodes(rows, task_id=task_id, bbox_rows=bbox_rows)
    return write_agibot(repo_root, episodes, task_id=task_id)
