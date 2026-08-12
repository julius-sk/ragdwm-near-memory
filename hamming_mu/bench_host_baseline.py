"""A 组基线:host 侧 Hamming 扫描 + top-k。

必须与设备测量在同一台机器上跑 —— 跨机器数字不可比。
用位打包 uint64 + np.bitwise_count(numpy >= 2.0);无该函数时退回查表法。
"""
import argparse
import time

import numpy as np

WORDS = 4


def scan_topk(sigs, query, k):
    x = np.bitwise_xor(sigs, query)
    if hasattr(np, "bitwise_count"):
        d = np.bitwise_count(x).sum(axis=1)
    else:
        d = np.unpackbits(x.view(np.uint8).reshape(-1, WORDS * 8), axis=1).sum(axis=1)
    idx = np.argpartition(d, k)[:k]
    return idx[np.argsort(d[idx], kind="stable")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=10_000_000)
    p.add_argument("-k", type=int, default=500)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    sigs = rng.integers(0, 2**64, size=(a.n, WORDS), dtype=np.uint64)
    query = rng.integers(0, 2**64, size=WORDS, dtype=np.uint64)

    scan_topk(sigs, query, a.k)   # warmup

    best = float("inf")
    for r in range(a.reps):
        t0 = time.perf_counter()
        scan_topk(sigs, query, a.k)
        ms = (time.perf_counter() - t0) * 1e3
        best = min(best, ms)
        print(f"  rep {r}: {ms:.3f} ms")

    gb = a.n * 32 / 1e9
    print(f"\nA 组 host 基线  N={a.n}  k={a.k}")
    print(f"  最小耗时      : {best:.3f} ms")
    print(f"  有效带宽      : {gb / (best / 1e3):.1f} GB/s")
    print(f"  需入主机的数据: {gb:.3f} GB")
    print(f"  numpy         : {np.__version__}  bitwise_count={hasattr(np, 'bitwise_count')}")


if __name__ == "__main__":
    main()
