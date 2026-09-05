#!/usr/bin/env python3
"""LeRobot v2.1 数据集后处理 Pipeline

数据集上传到训练服务器后、训练前运行，确保数据完整可用。

Pipeline 步骤:
  1. 结构检查    — 验证目录、元数据、文件数一致性
  2. 视频修复    — 全量解码扫描 mp4，检测并修复损坏视频
  3. 数据统计    — 生成数据集摘要（episode 数、帧数、相机、任务等）
  4. 报告输出    — JSON 报告保存到数据集目录

用法:
  # 完整 pipeline（检查 + 修复 + 统计）
  python pipeline.py /path/to/dataset

  # 仅检查，不修复
  python pipeline.py /path/to/dataset --check-only

  # 并行扫描（加速视频检查）
  python pipeline.py /path/to/dataset -j 8

  # 不保留 .bak 备份
  python pipeline.py /path/to/dataset --no-backup

  # 远程执行示例（通过 SSH 到江算）
  ssh gpu-master "ssh root@172.30.61.36 \\
    '/home/ubuntu/lerobot/.venv/bin/python3 \\
    /home/ubuntu/cigai_train/openpi-jiangsuan/scripts/data_postprocessing/pipeline.py \\
    /nvme2n1/dataset/<dataset_name> --fix --verify -j 4'"

依赖:
  - Python 3.8+
  - av (PyAV)
  - ffmpeg (系统命令)
  - pandas (可选，用于帧数统计)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 同目录下的子模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import structure_check
import video_repair
import data_stats


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║       LeRobot v2.1 数据集后处理 Pipeline                    ║
║       Data Post-Processing Pipeline                         ║
╚══════════════════════════════════════════════════════════════╝
"""


def run_pipeline(
    dataset_path: str,
    fix: bool = True,
    verify: bool = True,
    keep_backup: bool = True,
    workers: int = 4,
    verbose: bool = True,
) -> dict:
    """运行完整的数据后处理 Pipeline

    Args:
        dataset_path: 数据集路径
        fix: 是否自动修复损坏视频
        verify: 修复后是否验证
        keep_backup: 是否保留 .bak 备份
        workers: 并行扫描 worker 数
        verbose: 是否打印详细信息

    Returns:
        完整报告 dict
    """
    dataset_path = os.path.abspath(dataset_path)
    dataset_name = os.path.basename(dataset_path)

    if verbose:
        print(BANNER)
        print(f"  数据集: {dataset_name}")
        print(f"  路径:   {dataset_path}")
        print(f"  模式:   {'检查+修复' if fix else '仅检查'}")
        print(f"  并行:   workers={workers}")
        print()

    report = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": "1.0",
    }

    t_total = time.time()
    all_passed = True

    # ── Step 1: 结构检查 ─────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Step 1/3: 数据集结构 & 元数据检查")
        print("=" * 60)

    struct_result = structure_check.run(dataset_path, verbose=verbose)
    report["structure_check"] = struct_result

    if verbose:
        n_pass = sum(1 for c in struct_result["checks"] if c["ok"])
        n_fail = len(struct_result["checks"]) - n_pass
        status = "✅ PASS" if struct_result["ok"] else "❌ FAIL"
        print(f"  结果: {status}  ({n_pass} passed, {n_fail} failed)")

        # 打印失败项
        for c in struct_result["checks"]:
            if not c["ok"]:
                print(f"    ❌ {c['check']}: {c['detail']}")
        print()

    if not struct_result["ok"]:
        all_passed = False
        if verbose:
            print("  ⚠️  结构检查未通过，部分问题可能影响后续步骤")
            print()

    # ── Step 2: 视频完整性检查 & 修复 ────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Step 2/3: 视频完整性检查" + (" & 修复" if fix else ""))
        print("=" * 60)

    video_result = video_repair.run(
        dataset_path,
        fix=fix,
        verify=verify,
        keep_backup=keep_backup,
        workers=workers,
        verbose=verbose,
    )
    report["video_repair"] = {
        "ok": video_result["ok"],
        "total": video_result["total"],
        "corrupted": video_result["corrupted"],
        "fixed": video_result["fixed"],
        "fix_failed": video_result["fix_failed"],
        "scan_sec": video_result["scan_sec"],
        "repair_sec": video_result["repair_sec"],
    }

    if verbose:
        status = "✅ PASS" if video_result["ok"] else "❌ FAIL"
        print(f"  结果: {status}")
        print(f"    总计: {video_result['total']} 个视频")
        print(f"    损坏: {video_result['corrupted']}")
        if fix:
            print(f"    修复: {video_result['fixed']}")
            print(f"    失败: {video_result['fix_failed']}")
        print(f"    耗时: 扫描 {video_result['scan_sec']}s, 修复 {video_result['repair_sec']}s")
        print()

    if not video_result["ok"]:
        all_passed = False

    # ── Step 3: 数据统计 ─────────────────────────────────────────────
    if verbose:
        print("=" * 60)
        print("  Step 3/3: 数据集统计")
        print("=" * 60)

    stats_result = data_stats.run(dataset_path, verbose=verbose)
    report["data_stats"] = stats_result

    if verbose:
        print(f"  机器人: {stats_result.get('robot_type', '?')}")
        print(f"  Episodes: {stats_result.get('total_episodes', '?')}")
        print(f"  总帧数: {stats_result.get('total_frames', '?')}")
        print(f"  FPS: {stats_result.get('fps', '?')}")
        print(f"  相机: {', '.join(stats_result.get('cameras', []))}")
        print(f"  Action dim: {stats_result.get('action_dim', '?')}")
        print(f"  State dim: {stats_result.get('state_dim', '?')}")
        if "video_size" in stats_result:
            vs = stats_result["video_size"]
            print(f"  视频: {vs['count']} 个, 总计 {vs['total_mb']} MB")
        print(f"  数据集总大小: {stats_result.get('total_size_mb', '?')} MB")

        tasks = stats_result.get("tasks", [])
        if tasks:
            print(f"  任务 ({len(tasks)}):")
            for i, t in enumerate(tasks[:5]):
                print(f"    [{i}] {t[:80]}")
            if len(tasks) > 5:
                print(f"    ... 共 {len(tasks)} 个任务")
        print()

    # ── 总结 ─────────────────────────────────────────────────────────
    total_sec = round(time.time() - t_total, 1)
    report["all_passed"] = all_passed
    report["total_time_sec"] = total_sec

    if verbose:
        print("=" * 60)
        final = "✅ ALL PASSED" if all_passed else "⚠️  ISSUES FOUND"
        print(f"  Pipeline 完成: {final}  ({total_sec}s)")
        print("=" * 60)

    return report


def save_report(report: dict, dataset_path: str) -> str:
    """保存 JSON 报告到数据集目录"""
    report_dir = os.path.join(dataset_path, "meta")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "postprocessing_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="LeRobot v2.1 数据集后处理 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("dataset_path", help="数据集路径")
    parser.add_argument("--check-only", action="store_true",
                        help="仅检查，不修复")
    parser.add_argument("--fix", action="store_true", default=False,
                        help="自动修复损坏视频")
    parser.add_argument("--verify", action="store_true", default=False,
                        help="修复后验证")
    parser.add_argument("--no-backup", action="store_true",
                        help="不保留 .bak 备份")
    parser.add_argument("-j", "--workers", type=int, default=4,
                        help="并行扫描 worker 数 (default: 4)")
    parser.add_argument("--save-report", action="store_true", default=True,
                        help="保存 JSON 报告到数据集目录")
    parser.add_argument("--no-report", action="store_true",
                        help="不保存报告")

    args = parser.parse_args()

    # 检查数据集路径
    if not os.path.isdir(args.dataset_path):
        print(f"❌ 数据集路径不存在: {args.dataset_path}")
        sys.exit(1)

    # 确定模式
    fix = args.fix and not args.check_only
    verify = args.verify and not args.check_only

    # 运行 pipeline
    report = run_pipeline(
        dataset_path=args.dataset_path,
        fix=fix,
        verify=verify,
        keep_backup=not args.no_backup,
        workers=args.workers,
        verbose=True,
    )

    # 保存报告
    if not args.no_report:
        report_path = save_report(report, args.dataset_path)
        print(f"\n  📄 报告已保存: {report_path}")

    # 退出码
    sys.exit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
