"""gt.py —— 精确最近邻的地面真值(ground truth),流式计算,输出 big-ann-benchmarks 的 GT 格式。

GT 文件格式(与官方 *groundtruth*.bin 相同,可以直接互相比对):
  int32 nq, int32 k | nq x k int32 id | nq x k float32 距离(L2 平方)

为什么要自己算。官方 GT 只对应官方的规模(例如 DEEP1B 的 1B);我们要跑 10M / 100M 等
裁剪子集,裁剪后的最近邻和全集不同,必须重算。算法:基础集分块,每块用 FAISS 精确求 top-k,
再和已有结果按距离合并。--check 给出官方 GT 时,报告两者 top-k 的重合率,用来证明我们的
GT 代码本身是对的(先在官方规模上对一次,再信它算的子集)。

用法:
  python gt.py --base base.1B.fbin --crop 10000000 --queries query.public.10K.fbin --k 100 \
               --out gt/deep10m.gt100.bin
  python gt.py ... --crop 0 --check deep_new_groundtruth.public.10K.bin    # 对官方 GT 自检
"""
import argparse
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vecio  # noqa: E402


def read_gt(path):
    with open(path, "rb") as f:
        nq, k = np.fromfile(f, np.int32, 2)
        ids = np.fromfile(f, np.int32, nq * k).reshape(nq, k)
        dist = np.fromfile(f, np.float32, nq * k).reshape(nq, k)
    return ids, dist


def write_gt(path, ids, dist):
    with open(path, "wb") as f:
        np.array(ids.shape, np.int32).tofile(f)
        ids.astype(np.int32).tofile(f)
        dist.astype(np.float32).tofile(f)


def exact_knn(base, qv, k, chunk):
    nq = qv.shape[0]
    best_d = np.full((nq, k), np.inf, np.float32)
    best_i = np.full((nq, k), -1, np.int64)
    t0 = time.time()
    N = base.shape[0]
    for s, x in vecio.chunks(base, chunk):
        D, I = faiss.knn(qv, x, k)                 # L2 平方,精确
        I += s
        cd = np.concatenate([best_d, D], 1)
        ci = np.concatenate([best_i, I], 1)
        # 按 (距离, id) 排序:距离相同的取 id 小的,结果确定
        order = np.lexsort((ci, cd), axis=1)[:, :k]
        best_d = np.take_along_axis(cd, order, 1)
        best_i = np.take_along_axis(ci, order, 1)
        done = s + x.shape[0]
        el = time.time() - t0
        print("  GT %d / %d (%.0f s,预计还要 %.0f s)" % (done, N, el, el / done * (N - done)), flush=True)
    return best_i, best_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--column", default="emb")
    ap.add_argument("--qkey", default="test")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=1 << 23)
    ap.add_argument("--check", help="官方 GT 文件:报告重合率")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base = vecio.load(a.base, a.crop, key="train", column=a.column)
    qv = np.ascontiguousarray(vecio.load(a.queries, a.nq, key=a.qkey, column=a.column), np.float32)
    print("GT: N=%d d=%d nq=%d k=%d" % (base.shape[0], base.shape[1], qv.shape[0], a.k), flush=True)
    ids, dist = exact_knn(base, qv, a.k, a.chunk)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    write_gt(a.out, ids, dist)
    print("写出 %s" % a.out)
    if a.check:
        if a.check.endswith(".ivecs"):                 # TEXMEX 的官方 GT 只有 id,没有距离
            oi, od = np.asarray(vecio.load(a.check), np.int64), None
        else:
            oi, od = read_gt(a.check)
        kk = min(10, oi.shape[1], a.k)
        nq = min(oi.shape[0], ids.shape[0])
        ov = np.mean([len(set(ids[i, :kk]) & set(oi[i, :kk])) / kk for i in range(nq)])
        print("与官方 GT 对照(前 %d 名):id 重合 %.5f" % (kk, ov))
        if od is not None:
            # 距离并列会让 id 不同但同样正确,所以再报「第 k 名距离一致」
            rel = np.abs(dist[:nq, kk - 1] - od[:nq, kk - 1]) / np.maximum(od[:nq, kk - 1], 1e-12)
            print("  第 %d 名距离最大相对差 %.2e" % (kk, rel.max()))


if __name__ == "__main__":
    main()
