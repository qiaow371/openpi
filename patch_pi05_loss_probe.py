#!/usr/bin/env python3
"""给 openpi 的 PyTorch 训练脚本装「per-joint × per-timestep loss 探针」。

动机（2026-09-14 华为 16 NPU π0.5 事故复盘的第二层）：
日志里那个标量 ``loss`` 是 ``mean([B, H, D])``，把全部结构压扁了。"动作专家随机初始化"、
"某几维索引错位"、"夹爪维不学"、"左右臂反接"、"pad 维泄漏" 在标量曲线上一模一样 ——
都会正常下降。华为那次 loss 从 1.43 降到 0.005，看起来完全健康，真机才暴露问题。

本补丁在 ``loss.backward()`` 之前抓 ``losses`` 这个 ``[B, H, D]`` 未降维张量（不参与反传），
每个 log_interval 把 ``[H, D]`` 均值矩阵按行追加到 ``<checkpoint_dir>/loss_probe.jsonl``，
并在日志打一行紧凑摘要（人一眼看、LLM 也能直接读）。

指纹判读（D=32 的后 16 维是 PadStatesAndActions 补的 0）：pad 维目标 u = ε - 0 = ε，看似不可
预测，但 x_t = t·ε，给定 (x_t, t) 时 ε 完全可知 ⇒ 预训练好的 π0.5 早把 pad 维压到接近 0。
所以 **pad_mean ≈ 1 就是"动作专家从零开始学"的指纹，与任务难度无关**。

用法：
    python3 patch_pi05_loss_probe.py <path/to/scripts/train_pytorch.py> [--check]
幂等、纯文本插入；锚点不唯一即拒绝改动；先写临时文件，py_compile 通过才替换。
"""

from __future__ import annotations

import argparse
import os
import py_compile
import shutil
import sys

FUNC_ANCHOR = "def load_pytorch_weight_path("
BACKWARD_ANCHOR = "            # Backward pass\n            loss.backward()\n"
INFO_ANCHOR = (
    '                if ce_loss_v is not None:\n'
    '                    info["ce_loss"] = ce_loss_v\n'
    "                infos.append(info)\n"
)
LOG_ANCHOR = "                while loss_steps and loss_steps[-1] >= int(global_step):\n"

PROBE_BLOCK = '''
# ---------------------------------------------------------------------------
# per-joint × per-timestep loss 探针（2026-09-14 华为 π0.5 事故复盘）
# ---------------------------------------------------------------------------
# 标量 loss = mean([B, H, D]) 会掩盖所有结构性问题。这里保留 [H, D] 矩阵落盘。
# 判据：D=32 中后 16 维是 pad 0，其目标 u = ε，而 x_t = t·ε 使 ε 完全可知，
# 所以预训练好的 π0.5 pad 维 loss 已接近 0 —— pad_mean ≈ 1 等价于"动作专家是冷的"。
_PROBE_REAL_DIM = 16
_PROBE_PAD_MEAN_COLD = 0.5


def _loss_probe_stats(mat) -> dict:
    """[H, D] numpy 矩阵 → 可直接喂 LLM 的摘要 dict。"""
    real = mat[:, :_PROBE_REAL_DIM]
    pad = mat[:, _PROBE_REAL_DIM:]
    dim = real.mean(axis=0)
    return {
        "dim_real": [round(float(v), 5) for v in dim],
        "per_step": [round(float(v), 5) for v in mat.mean(axis=1)],
        "matrix": [[round(float(v), 5) for v in row] for row in mat],
        "real_mean": round(float(real.mean()), 5),
        "pad_mean": round(float(pad.mean()), 5),
        "worst_dim": int(dim.argmax()),
        "best_dim": int(dim.argmin()),
        "verdict": "cold-expert" if float(pad.mean()) > _PROBE_PAD_MEAN_COLD else "warm",
    }


def _write_loss_probe(checkpoint_dir: str, step, mat) -> None:
    """把 [H, D] 均值矩阵追加成一行 JSON。任何异常都只 warning —— 探针不许弄死训练。"""
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        path = os.path.join(checkpoint_dir, "loss_probe.jsonl")
        rec = {"step": int(step)}
        rec.update(_loss_probe_stats(mat))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\\n")
        logging.info(
            "PROBE step=%s real=%.4f pad=%.4f worst=d%d(%.3f) best=d%d(%.3f) verdict=%s",
            step,
            rec["real_mean"],
            rec["pad_mean"],
            rec["worst_dim"],
            rec["dim_real"][rec["worst_dim"]],
            rec["best_dim"],
            rec["dim_real"][rec["best_dim"]],
            rec["verdict"],
        )
    except Exception as e:  # noqa: BLE001 - 探针失败不能影响训练
        logging.warning("loss probe 写入失败（忽略）：%s", e)


'''


def patched_source(src: str) -> str:
    """返回打过补丁的源码；已打过则原样返回。"""
    if "_write_loss_probe" in src:
        return src

    for anchor, what in (
        (FUNC_ANCHOR, "load_pytorch_weight_path 函数定义"),
        (BACKWARD_ANCHOR, "backward 调用块"),
        (INFO_ANCHOR, "info 收集 / infos.append"),
        (LOG_ANCHOR, "loss 历史写入处"),
    ):
        if src.count(anchor) != 1:
            raise SystemExit(f"!! 锚点不唯一（找到 {src.count(anchor)} 处）：{what} —— 拒绝修改，请人工处理")
    if "\nimport os\n" not in src and not src.startswith("import os\n"):
        raise SystemExit("!! 找不到 import os —— 文件结构与预期不符，拒绝修改")
    if "\nimport numpy as np\n" not in src:
        raise SystemExit("!! 找不到 'import numpy as np' —— 文件结构与预期不符，拒绝修改")

    # 1) 探针函数放在加载函数之前（和 base guard 同一区域）
    src = src.replace(FUNC_ANCHOR, PROBE_BLOCK.lstrip("\n") + FUNC_ANCHOR, 1)

    # 2) 反传前抓未降维的 [B, H, D]
    src = src.replace(
        BACKWARD_ANCHOR,
        "            # per-joint × per-timestep 探针：不进反传，只取 batch 均值\n"
        "            fm_probe = None\n"
        "            if isinstance(losses, torch.Tensor) and losses.dim() == 3:\n"
        "                with torch.no_grad():\n"
        "                    fm_probe = losses.detach().float().mean(dim=0).cpu().numpy()  # [H, D]\n\n"
        + BACKWARD_ANCHOR,
        1,
    )

    # 3) 随 info 一起攒到 log_interval（复用已有的 infos 累积/重置机制）
    src = src.replace(
        INFO_ANCHOR,
        INFO_ANCHOR.replace(
            "                infos.append(info)\n",
            '                if fm_probe is not None:\n'
            '                    info["fm_probe"] = fm_probe\n'
            "                infos.append(info)\n",
        ),
        1,
    )

    # 4) 打点时求平均并落盘
    src = src.replace(
        LOG_ANCHOR,
        "                probe_mats = [i[\"fm_probe\"] for i in infos if \"fm_probe\" in i]\n"
        "                if probe_mats:\n"
        "                    _write_loss_probe(config.checkpoint_dir, global_step, sum(probe_mats) / len(probe_mats))\n"
        + LOG_ANCHOR,
        1,
    )
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
        print(f"OK 已含 loss 探针，无需改动：{args.target}")
        return 0
    if args.check:
        print(f"NEEDS-PATCH {args.target}")
        return 3

    backup = args.target + ".pre-lossprobe.bak"
    with open(backup, "w", encoding="utf-8") as f:
        f.write(src)
    tmp = args.target + ".new"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(new)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        print(f"!! 补丁后语法校验失败，原文件未动：{e}")
        os.remove(tmp)
        return 1
    shutil.move(tmp, args.target)
    print(f"PATCHED {args.target}  (备份 {backup})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
