#!/usr/bin/env python3
"""只读 safetensors 头部审计 π0.5 权重（不需要 torch，不吃内存，几 KB IO）。

用途：判断某个 model.safetensors 到底是「完整 π0.5」还是「只含 PaliGemma 的半成品」。
完整 π0.5 = 813 个张量（603 paligemma + 201 gemma_expert + 8 动作头/时间 MLP + tied embed_tokens）；
只含 PaliGemma 的半成品 = 604 个张量，缺 209 个动作侧张量，warm start 后动作专家随机初始化。

用法：
    python3 pi05_ckpt_header_check.py model.safetensors [--require-full]
退出码：0 正常 / 1 判定为不完整（或 --require-full 未满足） / 2 文件不可读
"""

from __future__ import annotations

import argparse
import collections
import json
import struct
import sys

ACTION_HEAD = ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out",
               "action_time_mlp_in", "action_time_mlp_out", "state_proj")
REQUIRED_PREFIXES = ("paligemma_with_expert.gemma_expert.", "action_in_proj.", "action_out_proj.")
MIN_EXPERT = 100
MIN_HEAD = 6


def classify(key: str) -> str:
    if key.startswith("paligemma_with_expert.paligemma."):
        return "paligemma(VLM)"
    if key.startswith("paligemma_with_expert.gemma_expert."):
        return "gemma_expert"
    if key.split(".")[0] in ACTION_HEAD:
        return "action_head"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--require-full", action="store_true", help="不完整则退出码 1（供脚本做门禁）")
    args = ap.parse_args()

    try:
        with open(args.path, "rb") as f:
            (head_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(head_len))
    except Exception as e:  # noqa: BLE001
        print(f"ERROR 读不到 safetensors 头部 {args.path}: {e}")
        return 2

    keys = [k for k in header if k != "__metadata__"]
    dist = collections.Counter(classify(k) for k in keys)
    dtypes = collections.Counter(header[k]["dtype"] for k in keys)
    absent = [p for p in REQUIRED_PREFIXES if not any(k.startswith(p) for k in keys)]
    full = dist["gemma_expert"] >= MIN_EXPERT and dist["action_head"] >= MIN_HEAD and not absent
    print(f"FILE={args.path}")
    print(f"KEYS={len(keys)}  DIST={dict(dist)}  DTYPES={dict(dtypes)}")
    print(f"VERDICT={'FULL' if full else 'INCOMPLETE'}" + (f"  缺失模块={absent}" if absent else ""))
    if not full:
        print("!! 该文件缺动作专家/动作投影，作为 warm start 基座会让动作头随机初始化；"
              " 而 strict=False 的加载器不会报错。")
    else:
        print("OK 动作专家与动作投影齐备，可作为 warm start 基座。")
    return 0 if (full or not args.require_full) else 1


if __name__ == "__main__":
    sys.exit(main())
