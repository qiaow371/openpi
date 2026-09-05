#!/usr/bin/env python3
"""把 CE 训好的 PaliGemma 拼回 π0.5 全模（保留原 Action Expert）。

输入:
  --base   pi05_fixed 目录（完整 model.safetensors）
  --pg     e2_subtask_ce/.../best（含 paligemma_with_expert.paligemma.*）
  --out    输出目录
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--pg", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = Path(args.base)
    pg = Path(args.pg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_f = base / "model.safetensors"
    pg_f = pg / "model.safetensors"
    if not base_f.is_file():
        raise SystemExit(f"missing {base_f}")
    if not pg_f.is_file():
        raise SystemExit(f"missing {pg_f}")

    prefix = "paligemma_with_expert.paligemma."
    with safe_open(str(base_f), framework="pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys()}
    n_base = len(state)

    replaced = 0
    added = 0
    with safe_open(str(pg_f), framework="pt") as f:
        for k in f.keys():
            key = k if k.startswith(prefix) else prefix + k
            if key in state:
                replaced += 1
            else:
                added += 1
            state[key] = f.get_tensor(k)

    save_file(state, str(out / "model.safetensors"))
    for name in ("tokenizer.model", "config.json", "assets"):
        src = base / name
        if src.is_file():
            shutil.copy2(src, out / name)
        elif src.is_dir():
            dst = out / name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    meta = {
        "kind": "pi05_spliced_paligemma",
        "base": str(base),
        "pg": str(pg),
        "n_base_keys": n_base,
        "replaced": replaced,
        "added": added,
        "n_out_keys": len(state),
    }
    (out / "splice_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
