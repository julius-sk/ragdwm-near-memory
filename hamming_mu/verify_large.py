"""大规模(N 上亿)的分块验证器 —— verify_host.py 的低内存替代。

为什么需要它:`verify_host.hamming_all` 用 `np.unpackbits`,会把 (N,32) uint8
展开成 (N,256) uint8。N=100M 时那是 25 GB 的中间数组,跑不动。

本文件改用 `np.bitwise_count` 并**分块**处理,峰值内存只有一个块的几倍,
所以 N=1e8 也能在几秒内验完。

求第 k 名距离用的是**和设备 kernel 相同的 257 桶计数排序** —— Hamming 距离是
[0,256] 的有界整数,所以一遍扫描累直方图就能定出精确阈值,无需排序、无需
把全部距离留在内存里。两侧用同一套逻辑,对称且互为佐证。

用法:
  # 生成大规模签名 + query(不预先算距离,省内存和磁盘)
  python verify_large.py gen -n 100000000 -o /tmp/sig100m.bin

  # 验证设备 dump 出来的 top-k ids
  python verify_large.py check-topk --sigs /tmp/sig100m.bin --ids /tmp/dev.ids -k 500

  # 验证设备 dump 出来的全部距离(B 模式)
  python verify_large.py check-dists --sigs /tmp/sig100m.bin --dists /tmp/dev.i32
"""

import argparse
import sys

import numpy as np

WORDS = 4                 # 每签名 4 x uint64 = 256 位
BITS = WORDS * 64         # 256
NBUCKETS = BITS + 1       # 距离 0..256
CHUNK = 1 << 20           # 每块 1M 个签名 = 32 MB,峰值内存约 100 MB


def _popcount_rows(x):
    """每行 4 个 uint64 的总 popcount。bitwise_count 不可用时退回查表。"""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x).sum(axis=1, dtype=np.int32)
    b = x.view(np.uint8).reshape(-1, WORDS * 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int32)


def iter_chunks(path, n=None):
    """分块读签名文件,yield (start_index, (m,4) uint64 数组)。"""
    mm = np.memmap(path, dtype="<u8", mode="r")
    total = mm.size // WORDS
    if n is not None:
        if n > total:
            sys.exit(f"FAIL: 请求 {n} 个签名,文件只有 {total} 个")
        total = n
    for start in range(0, total, CHUNK):
        end = min(start + CHUNK, total)
        yield start, np.asarray(mm[start * WORDS:end * WORDS]).reshape(-1, WORDS)


def read_query(path):
    q = np.fromfile(path + ".query", dtype="<u8", count=WORDS)
    if q.size != WORDS:
        sys.exit(f"FAIL: {path}.query 长度不对({q.size} != {WORDS})")
    return q


def build_histogram(sigs_path, query, n=None):
    """一遍扫描 -> 257 桶直方图 + 签名总数。与设备 kernel 同一套逻辑。"""
    hist = np.zeros(NBUCKETS, dtype=np.int64)
    total = 0
    for _start, block in iter_chunks(sigs_path, n):
        d = _popcount_rows(np.bitwise_xor(block, query))
        hist += np.bincount(d, minlength=NBUCKETS)
        total += block.shape[0]
    return hist, total


def kth_distance(hist, k):
    """第 k 名(1-indexed)的距离 = 累计计数首次 >= k 的那个桶。"""
    cum = np.cumsum(hist)
    idx = np.searchsorted(cum, k, side="left")
    if idx >= NBUCKETS:
        sys.exit(f"FAIL: 全部签名不足 {k} 个")
    return int(idx)


def distances_of(sigs_path, query, ids):
    """只算指定 id 的距离(随机读,数量少,直接取)。"""
    mm = np.memmap(sigs_path, dtype="<u8", mode="r")
    total = mm.size // WORDS
    bad = ids[(ids < 0) | (ids >= total)]
    if bad.size:
        sys.exit(f"FAIL: {bad.size} 个 id 越界(如 {int(bad[0])},文件只有 {total} 个签名)"
                 f" —— 设备可能返回了哨兵 0xFFFFFFFF,即收集不足 k 个")
    rows = np.stack([np.asarray(mm[int(i) * WORDS:(int(i) + 1) * WORDS]) for i in ids])
    return _popcount_rows(np.bitwise_xor(rows, query))


def cmd_gen(a):
    """分块生成并写盘,不预先算距离 —— 距离在 check 时按需分块算。"""
    rng = np.random.default_rng(a.seed)
    written = 0
    with open(a.out, "wb") as f:
        while written < a.n:
            m = min(CHUNK, a.n - written)
            blk = rng.integers(0, 2**64, size=(m, WORDS), dtype=np.uint64)
            f.write(np.ascontiguousarray(blk, dtype="<u8").tobytes())
            written += m
    q = rng.integers(0, 2**64, size=(1, WORDS), dtype=np.uint64)
    with open(a.out + ".query", "wb") as f:
        f.write(np.ascontiguousarray(q, dtype="<u8").tobytes())
    print(f"wrote {a.n} sigs -> {a.out} ({a.n * 32} B) + {a.out}.query")


def cmd_check_topk(a):
    query = read_query(a.sigs)
    ids = np.fromfile(a.ids, dtype=np.uint32)[:a.k]
    if ids.size != a.k:
        sys.exit(f"FAIL: 期望 {a.k} 个 id,文件里只有 {ids.size} 个")

    n_sent = int((ids == 0xFFFFFFFF).sum())
    if n_sent:
        sys.exit(f"FAIL: {n_sent} 个哨兵 0xFFFFFFFF —— 设备只收集到 {a.k - n_sent}/{a.k} 个真实 id")

    uniq, counts = np.unique(ids, return_counts=True)
    dup = uniq[counts > 1]
    if dup.size:
        sys.exit(f"FAIL: id {int(dup[0])} 出现 {int(counts[counts > 1][0])} 次(期望唯一)")

    hist, total = build_histogram(a.sigs, query, a.n)
    kth = kth_distance(hist, a.k)
    got = distances_of(a.sigs, query, ids)

    if got.max() > kth:
        sys.exit(f"FAIL: 返回了距离 {int(got.max())} 的 id,但第 {a.k} 名距离是 {kth}")
    print(f"PASS: N={total}  {a.k} 个 id 唯一且距离全部 <= 第k名({kth})"
          f";返回距离范围 [{int(got.min())}, {int(got.max())}]")


def cmd_check_dists(a):
    query = read_query(a.sigs)
    got = np.memmap(a.dists, dtype=np.int32, mode="r")
    pos = 0
    for _start, block in iter_chunks(a.sigs, a.n):
        ref = _popcount_rows(np.bitwise_xor(block, query))
        chunk = np.asarray(got[pos:pos + ref.size])
        if chunk.size != ref.size:
            sys.exit(f"FAIL: 距离文件长度不足(在偏移 {pos} 处)")
        bad = np.nonzero(chunk != ref)[0]
        if bad.size:
            i = int(bad[0])
            sys.exit(f"FAIL: idx={pos + i} ref={int(ref[i])} got={int(chunk[i])}"
                     f"(本块共 {bad.size} 处不符)")
        pos += ref.size
    if got.size != pos:
        sys.exit(f"FAIL: 距离文件有 {got.size} 项,签名只有 {pos} 个")
    print(f"PASS: {pos} 个距离逐位一致")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen"); g.set_defaults(fn=cmd_gen)
    g.add_argument("-n", type=int, required=True)
    g.add_argument("-o", "--out", required=True)
    g.add_argument("--seed", type=int, default=0)

    t = sub.add_parser("check-topk"); t.set_defaults(fn=cmd_check_topk)
    t.add_argument("--sigs", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("-k", type=int, required=True)
    t.add_argument("-n", type=int, default=None, help="只验证前 n 个签名(默认全部)")

    d = sub.add_parser("check-dists"); d.set_defaults(fn=cmd_check_dists)
    d.add_argument("--sigs", required=True)
    d.add_argument("--dists", required=True)
    d.add_argument("-n", type=int, default=None)

    args = p.parse_args()
    args.fn(args)
