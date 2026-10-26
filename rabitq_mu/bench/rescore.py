"""rescore.py —— 漏斗的最后一级:用原始浮点向量对设备返回的 k' 个候选重算精确距离,取前 k。

RaBitQ 论文和 FAISS 的常规用法都是「量化码筛出 k' 个候选,再读原始向量重排」。
设备一次最多返回 K_MAX = 64 个(kernel 常量),所以 k' <= 64。原始向量不在卡上:
DEEP1B 是 384 GB,在主机的数据盘上,这一步的代价就是每查询 k' 次随机读。

报告:重排前后的 recall@k,每查询读原始向量的字节数,以及(--time 时)冷读耗时。
冷读用 posix_fadvise(DONTNEED) 把文件页从主机页缓存赶走后再读;只在 Linux 上有意义。

用法:python rescore.py --dump dev_k64.bin --kp 64 --base base.1B.fbin --crop 100000000 \
                        --queries query.public.10K.fbin --gt gt.bin --k 10
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vecio                     # noqa: E402
from eval import read_dump, recall   # noqa: E402
from gt import read_gt           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--kp", type=int, required=True, help="dump 里每查询的候选数 k'")
    ap.add_argument("--base", required=True)
    ap.add_argument("--crop", type=int, default=0)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qkey", default="test")
    ap.add_argument("--column", default="emb")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--time", action="store_true", help="冷读计时(Linux)")
    ap.add_argument("--json")
    a = ap.parse_args()
    _, ids, _, _ = read_dump(a.dump, a.kp)
    nq = ids.shape[0]
    base = vecio.load(a.base, a.crop, key="train", column=a.column)
    qv = np.asarray(vecio.load(a.queries, nq, key=a.qkey, column=a.column), np.float32)[:nq]
    gi, gd = read_gt(a.gt)
    r0, _ = recall(ids[:, :a.k], gi[:nq], gd[:nq], a.k)
    if a.time and hasattr(os, "posix_fadvise") and isinstance(base, np.memmap):
        fd = os.open(base.filename, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    out_ids = np.empty((nq, a.k), np.int64)
    t0 = time.time()
    for i in range(nq):
        cand = ids[i][ids[i] < 0xFFFFFFFF]
        v = np.asarray(base[np.sort(cand)], np.float32)          # 排序后读,对磁盘友好一点
        dist = ((v - qv[i]) ** 2).sum(1)
        order = np.lexsort((np.sort(cand), dist))[:a.k]
        out_ids[i] = np.sort(cand)[order]
    el = time.time() - t0
    r1, _ = recall(out_ids, gi[:nq], gd[:nq], a.k)
    row_bytes = base.shape[1] * base.dtype.itemsize
    res = {"dump": a.dump, "kp": a.kp, "k": a.k, "recall_before": r0, "recall_after": r1,
           "bytes_per_query": a.kp * row_bytes, "ms_per_query": el / nq * 1e3 if a.time else None}
    print("重排前 recall@%d = %.4f(只用量化距离的前 %d 个)" % (a.k, r0, a.k))
    print("重排后 recall@%d = %.4f(读 %d 个原始向量,每查询 %.1f KB)"
          % (a.k, r1, a.kp, a.kp * row_bytes / 1e3))
    if a.time:
        print("冷读 + 重算:每查询 %.3f ms" % (el / nq * 1e3))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
