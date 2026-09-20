#!/usr/bin/env python3
"""给 openpi 的 PyTorch 训练脚本装「基座权重完整性硬校验」。

背景（2026-09-14 华为 16 NPU π0.5 事故）：
``load_pytorch_weight_path`` 用 ``strict=False`` 加载 warm start 基座，缺失 key 只打一行
warning。于是把只含 PaliGemma 的半成品（``pi05_fixed_paligemma/model.safetensors``，604 个
张量）当成完整基座（``pi05_fixed``，813 个张量）传上服务器时，209 个动作专家 / 动作投影张量
留在随机初始化状态，训练照跑、loss 照降，只有真机评估才暴露。

本脚本是幂等的、纯文本插入的补丁，可同时用于本机 fork 树和被手改过的服务器树：
    python3 patch_pi05_base_guard.py <path/to/scripts/train_pytorch.py> [--check]
插入后用 ``py_compile`` 验证语法；锚点不唯一或已存在则不动文件。
"""

from __future__ import annotations

import argparse
import py_compile
import sys

IMPORT_ANCHOR = "import platform\n"
FUNC_ANCHOR = "def load_pytorch_weight_path("
PATH_ANCHOR = '    model_path = os.path.join(weight_path, "model.safetensors")\n'
MISS_ANCHOR = '        if missing:\n            logging.warning("Missing weight keys (first 12): %s", list(missing)[:12])\n'

GUARD_BLOCK = '''
# ---------------------------------------------------------------------------
# 基座权重完整性硬校验（2026-09-14 华为 16 NPU π0.5 事故复盘）
# ---------------------------------------------------------------------------
# warm start 用 strict=False 是必要的：本地 vendor 的 PaliGemma 把 tied 权重挂在
# ``paligemma.model.language_model.embed_tokens.weight``，HF 的 PaliGemma 不注册该 key。
# 但 strict=False 也会把「传错 / 只传了一半」的基座静默吞掉：只含 PaliGemma 的
# ``pi05_fixed_paligemma/model.safetensors``（604 张量）相对完整 π0.5（813 张量）缺 209 个
# gemma_expert / action_in_proj / action_out_proj / time_mlp_* 张量 —— 动作专家随机初始化，
# 训练照样跑、train loss 照样降到 1e-3，只有真机评估才发现（华为效果差的根因）。
# 因此把容错收紧：缺失 key 只允许白名单，且基座必须真的含动作专家。

_BASE_MISSING_ALLOWLIST = ("language_model.embed_tokens.weight",)
_BASE_REQUIRED_PREFIXES = (
    "paligemma_with_expert.gemma_expert.",  # 动作专家主干
    "action_in_proj.",  # 动作/状态输入投影
    "action_out_proj.",  # 动作输出头
)


def _safetensors_keys(path: str) -> list[str]:
    """只读 safetensors 头部取 key 列表，不加载张量数据（头部一般几十 KB）。"""
    with open(path, "rb") as f:
        (head_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(head_len))
    return [k for k in header if k != "__metadata__"]


def _assert_base_complete(model_path: str, missing: list[str] | None = None) -> None:
    """校验基座确实含动作专家与动作投影，且 missing 只在白名单内，否则抛错中止训练。"""
    keys = _safetensors_keys(model_path)
    absent = [prefix for prefix in _BASE_REQUIRED_PREFIXES if not any(k.startswith(prefix) for k in keys)]
    if absent:
        raise RuntimeError(
            f"基座权重不完整：{model_path} 只有 {len(keys)} 个张量，完全不含 {absent}。"
            " 多半是误用了只含 PaliGemma 的半成品（如 pi05_fixed_paligemma），"
            " 继续训练会让动作专家随机初始化。请改用完整基座 pi05_fixed/。"
        )
    unknown = [k for k in (missing or []) if not any(allowed in k for allowed in _BASE_MISSING_ALLOWLIST)]
    if unknown:
        raise RuntimeError(
            f"基座权重缺少 {len(unknown)} 个非白名单张量（示例：{unknown[:5]}），"
            " 中止训练以免半随机初始化。"
        )
    logging.info("Base weight check OK: %s tensors, missing=%s (allowlist only)", len(keys), len(missing or []))


'''


def patched_source(src: str) -> str:
    """返回打过补丁的源码；已打过则原样返回。"""
    if "_assert_base_complete" in src:
        return src

    for anchor, what in ((FUNC_ANCHOR, "load_pytorch_weight_path 函数定义"), (PATH_ANCHOR, "model_path 赋值行"), (MISS_ANCHOR, "missing 日志块")):
        if src.count(anchor) != 1:
            raise SystemExit(f"!! 锚点不唯一（找到 {src.count(anchor)} 处）：{what} —— 拒绝修改，请人工处理")
    if IMPORT_ANCHOR not in src:
        raise SystemExit("!! 找不到 import 锚点 'import platform' —— 文件结构与预期不符，拒绝修改")

    # 1) 补依赖导入
    extra = "import json\n" if "\nimport json\n" not in src and not src.startswith("import json\n") else ""
    extra += "import struct\n" if "\nimport struct\n" not in src and not src.startswith("import struct\n") else ""
    if extra:
        src = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + extra, 1)

    # 2) 在函数前插入校验工具
    src = src.replace(FUNC_ANCHOR, GUARD_BLOCK.lstrip("\n") + FUNC_ANCHOR, 1)

    # 3) 调用点 A：加载前先做文件级检查（失败最快、且不依赖加载结果）
    src = src.replace(PATH_ANCHOR, PATH_ANCHOR + "    _assert_base_complete(model_path)\n", 1)

    # 4) 调用点 B：加载后按白名单核对 missing key
    src = src.replace(MISS_ANCHOR, MISS_ANCHOR + "        _assert_base_complete(model_path, missing)\n", 1)
    return src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="scripts/train_pytorch.py 的路径")
    ap.add_argument("--check", action="store_true", help="只报告是否需要/可应用补丁，不写文件")
    args = ap.parse_args()

    with open(args.target, encoding="utf-8") as f:
        src = f.read()
    new = patched_source(src)

    if new == src:
        print(f"OK 已含基座完整性校验，无需改动：{args.target}")
        return 0
    if args.check:
        print(f"NEEDS-PATCH {args.target}")
        return 3

    backup = args.target + ".pre-baseguard.bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(src)
    tmp = args.target + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"!! 补丁后语法校验失败，原文件未动：{e}")
        import os as _os

        _os.remove(tmp)
        return 1
    import os as _os
    import shutil as _sh

    _sh.move(tmp, args.target)
    print(f"PATCHED {args.target}  (备份 {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
