#!/usr/bin/env python3
"""把 fp32 的 π0.5 safetensors 基座压成 bf16（体积减半），并保持与模型参数 dtype 一致。

openpi 的 PyTorch 路径里 `PaliGemmaWithExpertModel` 会 `self.to(bfloat16)`，但把少数参数
留在 fp32（SigLIP patch/position embedding 与各 layernorm）。本脚本按同一份白名单决定每个
张量的目标 dtype，因此落盘文件与 `load_state_dict` 的实际拷贝结果完全一致。

用法：
    python3 cast_safetensors_bf16.py IN.safetensors OUT.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys

import torch
from safetensors.torch import load_file, save_file

# 与 src/openpi/models_pytorch/gemma_pytorch.py: params_to_keep_float32 保持一致
KEEP_FP32 = (
    "vision_tower.vision_model.embeddings.patch_embedding.weight",
    "vision_tower.vision_model.embeddings.patch_embedding.bias",
    "vision_tower.vision_model.embeddings.position_embedding.weight",
    "input_layernorm",
    "post_attention_layernorm",
    "model.norm",
)
# PI0Pytorch 顶层的动作投影/时间 MLP 是普通 nn.Linear（fp32），只有 paligemma_with_expert
# 子模块被 cast 成 bf16 —— 这些 key 必须保持 fp32，否则 warm start 会白白掉精度。
BF16_SCOPE = "paligemma_with_expert."
REQUIRED_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",
    "action_in_proj.",
    "action_out_proj.",
)


def target_dtype(key: str) -> "torch.dtype":
    """复刻模型实际参数 dtype：paligemma_with_expert 内 bf16（白名单除外），其余 fp32。"""
    if not key.startswith(BF16_SCOPE):
        return torch.float32
    if any(sel in key for sel in KEEP_FP32):
        return torch.float32
    return torch.bfloat16


def load_header(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def header_keys(path: str) -> list[str]:
    return [k for k in load_header(path) if k != "__metadata__"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--reference", help="同架构的真实 ckpt，用于逐张量对照 dtype")
    args = ap.parse_args()

    keys = header_keys(args.src)
    absent = [p for p in REQUIRED_PREFIXES if not any(k.startswith(p) for k in keys)]
    if absent:
        print(f"!! 源文件本身就不完整，缺 {absent}：{args.src}")
        return 1
    print(f"源：{len(keys)} 个张量")

    state = load_file(args.src)
    out: dict[str, torch.Tensor] = {}
    n_fp32 = 0
    for k in keys:
        t = state.pop(k)
        want = target_dtype(k)
        v = t.detach().to(want).contiguous().clone()
        n_fp32 += want == torch.float32
        out[k] = v
    del state
    save_file(out, args.dst, metadata={"format": "pt"})
    print(f"目标：{len(out)} 个张量，其中 {n_fp32} 个保持 fp32（动作头 + layernorm / vision embedding）")

    # 自校验：key 一致、dtype 合理、无 NaN
    assert header_keys(args.dst) == sorted(keys) or set(header_keys(args.dst)) == set(keys), "key 集合不一致"
    chk = load_file(args.dst)
    bad = [k for k, t in chk.items() if t.dtype not in (torch.bfloat16, torch.float32)]
    assert not bad, f"意外 dtype：{bad[:3]}"
    probe = chk["action_out_proj.weight"].float()
    assert torch.isfinite(probe).all(), "action_out_proj 含 NaN/Inf"
    g = "gemma_expert" if any(k.startswith(REQUIRED_PREFIXES[0]) for k in chk) else "MISSING"
    print(f"校验通过：{len(chk)} 张量，gemma_expert={g}，action_out_proj 有限值 ✓")

    # 与真实训练 ckpt 对照 per-key dtype（同一套模型代码产出的 ckpt 就是标准答案）
    if args.reference:
        ref = {k: v["dtype"] for k, v in load_header(args.reference).items() if k != "__metadata__"}
        new = {k: v["dtype"] for k, v in load_header(args.dst).items() if k != "__metadata__"}
        common = set(ref) & set(new)
        mism = [(k, ref[k], new[k]) for k in sorted(common) if ref[k] != new[k]]
        print(f"对照 ckpt：共有 {len(common)} 个 key，dtype 不一致 {len(mism)} 个")
        for m in mism[:10]:
            print("   不一致:", m)
        print(f"   只在参考 ckpt: {sorted(set(ref) - set(new))[:5]}")
        print(f"   只在新文件: {sorted(set(new) - set(ref))[:5]}")
        if mism:
            print("!! dtype 与真实 ckpt 不一致，说明目标 dtype 规则没对上模型")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
