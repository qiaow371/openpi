#!/usr/bin/env python3
"""在服务器端把「上传的缺失张量」+「服务器已有的 PaliGemma 张量」拼回逐字节相同的完整 π0.5 基座。

配套 `pi05_base_split.py` 使用，解决的是单向窄带问题：华为机上行 ~0.2 MiB/s，整包 8.53 GiB 要 12 小时，
但它那份坏基座里 564 颗 BF16 PaliGemma 张量与完整基座同源（6.42 GiB），逐颗 md5 能对上就不传，
实传只有 249 颗 / 1.52 GiB。

关键设计：**按上传的原始头部重建**，输出文件的头部与数据排布和源文件完全一致，
所以产物 md5 == manifest.source_md5 == 本地整文件 md5 —— 下游 `huawei_fixbase_apply.sh`
的 md5 / 头部张量数门禁一行都不用改。

用法：
  # 1) 只校验服务器那份能不能复用，不写盘（驱动据此决定走快路还是回落整文件上传）
  python3 pi05_base_merge.py --part PARTDIR --theirs /home/pi05/models/model.safetensors --precheck
  # 2) 流式重建 + 校验 + 原子落位（峰值内存 ~16 MiB）
  python3 pi05_base_merge.py --part PARTDIR --theirs ... --write --out /tmp/x --finalize STAGE

补传：--precheck 若发现某些"该复用"的张量在服务器那份里对不上（缺失或字节不同），退出码 3 并把
清单写进 --report；本地用 `pi05_base_split.py --send-keys-from 清单 --outdir PARTDIR2` 只传这些，
再 `--part PARTDIR --part PARTDIR2` 一起合并即可，不必重传 6.4 GiB。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys

CHUNK = 1 << 24  # 16 MiB


def md5_of(fh, start: int, length: int) -> str:
    h = hashlib.md5()
    fh.seek(start)
    left = length
    while left > 0:
        b = fh.read(min(CHUNK, left))
        if not b:
            raise SystemExit(f"读越界：需要 {length} 字节 @ {start}")
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def read_part(part: pathlib.Path) -> tuple[dict, bytes]:
    """读一个 split 产物目录，返回 (manifest, header_bytes)。"""
    for name in ("manifest.json", "header.bin"):
        if not (part / name).is_file():
            raise SystemExit(f"{part} 缺 {name}（先跑 pi05_base_split.py）")
    man = json.loads((part / "manifest.json").read_text())
    head = (part / "header.bin").read_bytes()
    if len(head) != 8 + man["header_len"]:
        raise SystemExit(f"{part}/header.bin 长度 {len(head)} != 8+{man['header_len']}，头部被截断")
    for key in ("bundle.bin", "bundle.md5"):
        if man["send_keys"] and not (part / key).is_file():
            raise SystemExit(f"{part} 缺 {key}（有 {len(man['send_keys'])} 颗待传张量）")
    return man, head


def stream(dst, src_fh, src_off: int, length: int) -> None:
    src_fh.seek(src_off)
    left = length
    while left > 0:
        b = src_fh.read(min(CHUNK, left))
        if not b:
            raise SystemExit(f"读越界：{length} 字节 @ {src_off}")
        dst.write(b)
        left -= len(b)


def check_coverage(header: dict) -> int:
    """确认 data 区无空洞/无重叠，返回数据总字节数。"""
    spans = sorted((v["data_offsets"][0], v["data_offsets"][1])
                   for k, v in header.items() if k != "__metadata__")
    pos = 0
    for a, b in spans:
        if a != pos:
            raise SystemExit(f"data 区不连续：期望从 {pos} 开始，实际 {a}（本脚本假定整文件无空洞）")
        pos = b
    return pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", action="append", required=True, type=pathlib.Path,
                    help="pi05_base_split.py 的 --outdir，可重复（多批补传）")
    ap.add_argument("--theirs", required=True, help="服务器上现有的（坏的）基座，用来复用张量")
    ap.add_argument("--precheck", action="store_true", help="只逐颗校验可复用性，不写输出")
    ap.add_argument("--write", action="store_true", help="流式重建完整基座")
    ap.add_argument("--out", default=None, help="--write 时的输出路径（临时名）")
    ap.add_argument("--finalize", default=None, help="产物 md5 校验通过后 os.replace 到该路径")
    ap.add_argument("--report", default="reuse_fail.txt", help="不可复用张量清单，喂给 split --send-keys-from")
    args = ap.parse_args()

    if not (args.precheck or args.write):
        raise SystemExit("要指定 --precheck 和/或 --write")

    parts = [read_part(p) for p in args.part]
    man0, head0 = parts[0]
    header = json.loads(head0[8:])
    keys = [k for k in header if k != "__metadata__"]
    if len(keys) != 813:
        raise SystemExit(f"头部有 {len(keys)} 个张量，不是完整 π0.5（期望 813），拒绝合并")
    for p, (man, head) in zip(args.part, parts):
        if head != head0 or man["source_md5"] != man0["source_md5"]:
            raise SystemExit(f"{p} 的头部/源文件与 {args.part[0]} 不一致，不能一起合并")

    # 每颗张量的字节来源：某个 part 的 bundle，或服务器那份
    sent: dict[str, tuple[pathlib.Path, int]] = {}
    for p, (man, _) in zip(args.part, parts):
        base = 0
        for k in man["send_keys"]:
            sent[k] = (p / "bundle.bin", base)
            base += man["tensors"][k]["len"]
    reuse = [k for k in keys if k not in sent]
    send_bytes = sum(man0["tensors"][k]["len"] for k in sent)
    reuse_bytes = sum(man0["tensors"][k]["len"] for k in reuse)
    print(f"合并计划：{len(sent)} 颗来自上传件（{send_bytes / 2**30:.3f} GiB） + "
          f"{len(reuse)} 颗复用 {args.theirs}（{reuse_bytes / 2**30:.3f} GiB）")
    total_data = check_coverage(header)
    if total_data != man0["source_bytes"] - man0["data_start"]:
        raise SystemExit(f"data 区 {total_data} != 源文件 {man0['source_bytes']} - 头部 {man0['data_start']}")

    theirs = pathlib.Path(args.theirs)
    if not theirs.is_file():
        raise SystemExit(f"找不到 {theirs}")
    with open(theirs, "rb") as f:
        (their_head_len,) = struct.unpack("<Q", f.read(8))
        their_header = json.loads(f.read(their_head_len))
        their_data = 8 + their_head_len
        bad: list[str] = []
        for i, k in enumerate(sorted(reuse), 1):
            v = their_header.get(k)
            if v is None:
                bad.append(k)
            elif md5_of(f, their_data + v["data_offsets"][0],
                        v["data_offsets"][1] - v["data_offsets"][0]) != man0["tensors"][k]["md5"]:
                bad.append(k)
            if i % 100 == 0 or i == len(reuse):
                print(f"  校验 {i}/{len(reuse)} 颗，不可复用 {len(bad)}", flush=True)

    if bad:
        pathlib.Path(args.report).write_text("\n".join(bad) + "\n", encoding="utf-8")
        rb = sum(man0["tensors"][k]["len"] for k in bad)
        print(f"!! {len(bad)} 颗复用失败（缺失或字节不同，合计 {rb / 2**30:.3f} GiB），清单 → {args.report}")
        print(f"   补传：python3 pi05_base_split.py <完整基座> --outdir PART2 --send-keys-from {args.report}")
        print(f"   然后：pi05_base_merge.py --part {' --part '.join(str(p) for p in args.part)} --part PART2 ...")
        return 3
    print(f"OK {len(reuse)} 颗可复用，逐颗 md5 全部命中")
    if args.precheck and not args.write:
        return 0

    if not args.out:
        raise SystemExit("--write 需要 --out")
    out = pathlib.Path(args.out)
    if out.exists():
        out.unlink()
    bundle_fh: dict[pathlib.Path, object] = {}
    with open(theirs, "rb") as tf, open(out, "wb") as w:
        w.write(head0)
        for k, v in sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                           key=lambda kv: kv[1]["data_offsets"][0]):
            n = v["data_offsets"][1] - v["data_offsets"][0]
            if k in sent:
                path, off = sent[k]
                fh = bundle_fh.get(path)
                if fh is None:
                    fh = bundle_fh[path] = open(path, "rb")
                stream(w, fh, off, n)
            else:
                tv = their_header[k]
                stream(w, tf, their_data + tv["data_offsets"][0], n)
        w.flush()
    for fh in bundle_fh.values():
        fh.close()

    got = out.stat().st_size
    want = man0["source_bytes"]
    if got != want:
        raise SystemExit(f"!! 产物 {got} 字节 != 期望 {want}")
    h = hashlib.md5()
    with open(out, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    print(f"产物 md5={h.hexdigest()}  期望={man0['source_md5']}")
    if h.hexdigest() != man0["source_md5"]:
        raise SystemExit("!! md5 不一致，产物留在 " + str(out) + " 供排查，未落位")
    print("OK md5 与本地完整基座逐字节一致")
    if args.finalize:
        os.replace(out, args.finalize)
        print(f"已落位 → {args.finalize}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
