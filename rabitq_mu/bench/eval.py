"""eval.py —— 读设备(rabitq_scan --mode T/G)或主机(host_rabitq / faiss_host.py)的 dump,
对照地面真值算召回,并可把两份 dump 逐 id 互相对照。

dump 格式(三者相同):每查询 k x (float32 距离, uint32 id) + uint64 精算次数 + uint64 处理条数。

召回的定义沿用 RaBitQ 论文和 big-ann-benchmarks:recall@k = |返回的 k 个 ∩ 精确前 k 个| / k,
对全部查询取平均。「精确前 k 个」来自 gt.py 或官方 GT(按原始浮点向量的 L2 距离)。
距离并列时 id 集合不唯一:big-ann 的做法是把与第 k 名距离相同的 id 都算作正确,这里同样处理。

用法:
  python eval.py --dump dev.bin --gt deep10m.gt100.bin --k 10
  python eval.py --dump dev.bin --other host.bin --k 10       # 设备 vs 主机逐 id 对照
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gt import read_gt  # noqa: E402


def read_dump(path, k):
    rec = np.dtype([("d", "<f4", (k,)), ("i", "<u4", (k,)), ("ref", "<u8"), ("proc", "<u8")])
    raw = np.fromfile(path, np.uint8)
    per = k * 8 + 16
    if raw.size % per:
        sys.exit("%s 大小 %d 不是 %d 的整数倍:k 不对?" % (path, raw.size, per))
    nq = raw.size // per
    body = raw.reshape(nq, per)
    pairs = body[:, :k * 8].copy().view("<u4").reshape(nq, k, 2)
    dist = pairs[:, :, 0].copy().view("<f4")
    ids = pairs[:, :, 1].astype(np.int64)
    tail = body[:, k * 8:].copy().view("<u8")
    return dist, ids, tail[:, 0], tail[:, 1]


def recall(ids, gt_ids, gt_dist, k):
    nq = ids.shape[0]
    hits = []
    for i in range(nq):
        kth = gt_dist[i, k - 1]
        ok = set(gt_ids[i, :k].tolist())
        # 并列:与第 k 名同距离的 GT 条目也算对(GT 文件里 k 往后还有更多列时)
        extra = gt_ids[i, k:][gt_dist[i, k:] <= kth * (1 + 1e-6)]
        ok.update(extra.tolist())
        hits.append(len(ok & set(ids[i, :k].tolist())) / k)
    return float(np.mean(hits)), np.asarray(hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--k", type=int, default=10, help="dump 里每查询的 k(= 参数块里的 k)")
    ap.add_argument("--gt")
    ap.add_argument("--other", help="另一份 dump:逐 id 对照")
    ap.add_argument("--N", type=int, default=0, help="库大小,用于报告精算比例")
    ap.add_argument("--json", help="把结果追加写成一行 JSON")
    a = ap.parse_args()
    dist, ids, ref, proc = read_dump(a.dump, a.k)
    nq = ids.shape[0]
    out = {"dump": a.dump, "nq": int(nq), "k": a.k,
           "refined_mean": float(ref.mean()), "refined_median": float(np.median(ref))}
    N = a.N or int(proc.max())
    out["refined_frac"] = float(ref.mean() / max(N, 1))
    print("%s: %d 个查询, 精算 平均 %.0f / 中位 %.0f 个 (%.4f%% of N=%d)"
          % (a.dump, nq, ref.mean(), np.median(ref), 100 * out["refined_frac"], N))
    if (ids < 0).any() or (ids >= 0xFFFFFFFF).any():
        print("  !! 有空槽(id = NONE):返回不足 k 个")
    if a.gt:
        gi, gd = read_gt(a.gt)
        if gi.shape[0] < nq:
            sys.exit("GT 只有 %d 个查询,dump 有 %d 个" % (gi.shape[0], nq))
        gi, gd = gi[:nq], gd[:nq]
        for kk in sorted({1, a.k}):
            if kk <= a.k and kk <= gi.shape[1]:
                r, _ = recall(ids, gi, gd, kk)
                out["recall@%d" % kk] = r
                print("  recall@%d = %.4f(对照精确 L2 前 %d 名)" % (kk, r, kk))
    if a.other:
        d2, i2, ref2, _ = read_dump(a.other, a.k)
        m = min(nq, i2.shape[0])
        same = np.mean([set(ids[i].tolist()) == set(i2[i].tolist()) for i in range(m)])
        rel = np.abs(dist[:m] - d2[:m]) / np.maximum(np.abs(d2[:m]), 1e-12)
        out["same_set_frac"] = float(same)
        out["max_rel_dist_diff"] = float(rel.max())
        print("  与 %s 对照:top-%d id 集合完全相同的查询 %.4f,排序后距离最大相对差 %.2e"
              % (a.other, a.k, same, rel.max()))
        print("  精算次数:本 dump 平均 %.0f,对方 %.0f" % (ref.mean(), ref2[:m].mean()))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
