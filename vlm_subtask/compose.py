"""FSM 标签 → π0.5 短 subtask，并按 episode 切 train/val。

从 annotate_pipeline 拷贝，供 OpenPI π0.5 单独使用。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

FROZEN_DIR = "splits_frozen"
FROZEN_MARKER = "FROZEN"


def _hand_phrase(hand: str) -> str:
    if hand == "left":
        return "left"
    if hand == "right":
        return "right"
    return "arm"


def render_subtask(label: str, hand: str, templates: dict[str, str]) -> str:
    tmpl = templates.get(label)
    if not tmpl:
        raise KeyError(f"configs 缺少 subtask_templates.{label}")
    return tmpl.format(hand=_hand_phrase(hand)).strip()


def _episode_split(episodes: list[int], val_ratio: float, seed: int) -> tuple[set[int], set[int]]:
    uniq = sorted(set(episodes))
    if len(uniq) < 2:
        return set(uniq), set()
    n_val = max(1, int(round(len(uniq) * val_ratio)))
    scored = [(hashlib.md5(f"{seed}:{ep}".encode()).hexdigest(), ep) for ep in uniq]
    scored.sort()
    val = {ep for _, ep in scored[:n_val]}
    train = set(uniq) - val
    if not train:
        train, val = val, set()
    return train, val


def compose_rows(
    segs: list[dict],
    *,
    task: str,
    templates: dict[str, str],
    verify_rows: list[dict] | None = None,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    label_fix: dict[tuple[int, int, int], str] = {}
    for v in verify_rows or []:
        if v.get("verdict") == "disagree" and v.get("correct_label"):
            key = (int(v["episode"]), int(v["t0"]), int(v["t1"]))
            label_fix[key] = str(v["correct_label"])

    rows: list[dict] = []
    for seg in segs:
        ep = int(seg["episode"])
        t0, t1 = int(seg["t0"]), int(seg["t1"])
        label = label_fix.get((ep, t0, t1), str(seg["label"]))
        hand = str(seg.get("hand") or "none")
        subtask = render_subtask(label, hand, templates)
        rows.append(
            {
                "dataset_id": seg.get("dataset_id"),
                "episode": ep,
                "t0": t0,
                "t1": t1,
                "n_frames": int(seg.get("n_frames") or (t1 - t0)),
                "fsm_label": str(seg["label"]),
                "label": label,
                "hand": hand,
                "task": task.strip(),
                "subtask": subtask,
                "sheet": seg.get("sheet"),
                "sample_type": "high_level",
            }
        )

    train_eps, val_eps = _episode_split([r["episode"] for r in rows], val_ratio, seed)
    for r in rows:
        r["split"] = "val" if r["episode"] in val_eps else "train"

    summary = {
        "n": len(rows),
        "train": sum(r["split"] == "train" for r in rows),
        "val": sum(r["split"] == "val" for r in rows),
        "train_episodes": sorted(train_eps),
        "val_episodes": sorted(val_eps),
        "labels": dict(Counter(r["label"] for r in rows)),
        "label_fixes": len(label_fix),
    }
    return rows, summary


def write_splits(out: Path, rows: list[dict], summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def dump(name: str, items: list[dict]) -> None:
        with (out / name).open("w") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump("cot.jsonl", rows)
    dump("vlm_train.jsonl", [r for r in rows if r["split"] == "train"])
    dump("vlm_val.jsonl", [r for r in rows if r["split"] == "val"])
    (out / "compose_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    freeze_splits(out)


def freeze_splits(out: Path) -> Path:
    """First compose wins: keep the episode-level train/val split for VLM curves."""
    frozen = out / FROZEN_DIR
    marker = frozen / FROZEN_MARKER
    if marker.is_file():
        return frozen
    frozen.mkdir(parents=True, exist_ok=True)
    for name in ("vlm_train.jsonl", "vlm_val.jsonl", "cot.jsonl", "compose_summary.json"):
        src = out / name
        if src.is_file():
            shutil.copy2(src, frozen / name)
    marker.write_text("train/val episode split locked; do not overwrite\n")
    return frozen


def split_jsonl_dir(data_dir: Path) -> Path:
    frozen = data_dir / FROZEN_DIR
    if (frozen / "vlm_train.jsonl").is_file():
        return frozen
    return data_dir


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
