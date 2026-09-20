#!/usr/bin/env python3
"""逐张量 sha256 摘要 safetensors（只依赖 stdlib，不需要 torch，不吃内存）。

用途：证明「服务器上现用的基座」与「本机已验证的基座」逐张量比特一致 —— 换基座来源
（HF 直下 + cast / 隧道上传）时，用这个把「来源不同」变成可判定的等价性检查，而不是
只看张量个数。

    python3 safetensors_tensor_hashes.py IN.safetensors --out a.txt
    python3 safetensors_tensor_hashes.py --diff a.txt b.txt

diff 模式允许「只在一侧出现」的 key（如本地多出 tied embed_tokens），但内容不一致即失败。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys


def header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)), 8 + n


def digest(path: str, out=sys.stdout) -> int:
    h, base = header(path)
    keys = sorted(k for k in h if k != "__metadata__")
    for k in keys:
        m = h[k]
        s, e = m["data_offsets"]
        hh = hashlib.sha256()
        with open(path, "rb") as f:
            f.seek(base + s)
            left = e - s
            while left > 0:
                buf = f.read(min(1 << 22, left))
                if not buf:
                    break
                hh.update(buf)
                left -= len(buf)
        print(f"{k}\t{m['dtype']}\t{','.join(map(str, m['shape']))}\t{hh.hexdigest()}", file=out)
    print(f"# {path}: {len(keys)} tensors", file=sys.stderr)
    return 0


def diff(a: str, b: str) -> int:
    def load(p: str) -> dict:
        d = {}
        with open(p) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                k, dt, sh, sha = line.rstrip("\n").split("\t")
                d[k] = (dt, sh, sha)
        return d

    A, B = load(a), load(b)
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    bad = [k for k in sorted(set(A) & set(B)) if A[k] != B[k]]
    print(f"共同 key={len(set(A) & set(B))}，只在 A={len(only_a)}，只在 B={len(only_b)}，内容不一致={len(bad)}")
    for k in only_a[:5]:
        print(f"  只在 A: {k}")
    for k in only_b[:5]:
        print(f"  只在 B: {k}")
    for k in bad[:10]:
        print(f"  不一致: {k}\n    A={A[k]}\n    B={B[k]}")
    if bad:
        print("!! 存在逐张量不一致，两个基座不是同一份权重")
        return 1
    print("OK 共有张量逐比特一致；差异只在单侧独有的 key")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--out", default="")
    ap.add_argument("--diff", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()
    if args.diff:
        return diff(*args.diff)
    if not args.path:
        ap.error("需要 path 或 --diff A B")
    if args.out:
        with open(args.out, "w") as f:
            return digest(args.path, f)
    return digest(args.path)


if __name__ == "__main__":
    sys.exit(main())
