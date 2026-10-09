"""scale_kprime.py —— 把 RAGDWM 在四个小语料上的两个结论推到 10 万 / 100 万 / 884 万条。

开发机上的结论(语料最大 5.7 万条,RAGDWM/reports/rabitq_vs_simhash.json、rabitq_refine.json):
  1. 要让 95% 的查询保住精确 top-10,SimHash 需要的候选数随库增大而增大,
     RaBitQ 几乎不变(FiQA 上 SimHash 384 位 6,435 个,RaBitQ 4 bit 15 个)
  2. 真实 top-10 两阶段搜索里,第二阶段精算的比例随库增大而下降(FiQA 0.75%)
本脚本在同一模型(bge-small)、同一指标下检验它们在大一到两个数量级的库上是否成立。

每个规模 N:
  精确 top-10           FAISS IndexFlatIP(向量已归一化,内积 = cosine)
  SimHash 384 / 768 位  K'_min = #{Hamming <= 真 top-10 中最大的 Hamming}(含打平,与开发机一致)
  RaBitQ 1 / 2 / 4 bit  K'_min = #{码估计距离 <= 真 top-10 中最大的估计距离}
                        nb=1 用第一阶段估计,nb>=2 用多 bit 完整距离 —— 与开发机实验相同
  精算次数(nb=2, 4)    重放 FAISS 的两阶段扫描,task 数 1 与 2816

884 万条时不能像开发机那样用 FAISS 的 k=N 搜索拿全序(每查询要排 884 万个),
所以距离用分块矩阵乘法批量算。算法与 rabitq_ref.py 相同,每个 N 的第一块都会
与参考模型逐项比对,不一致就停。

用法:
  python scale_kprime.py --emb ../emb_msmarco --sizes 100000,1000000,0 --nq 1000 --nq-refine 100
  (--sizes 中 0 表示全部;子集抽样与 prep_rabitq.py --subset 相同)
"""
import argparse
import json
import os
import sys
import time

import faiss
import numpy as np
from numba import njit, prange

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rabitq_ref as R      # noqa: E402
import vecdata              # noqa: E402

K = 10
SUCCESS = 0.95
CHUNK = 200_000


# ---------------------------------------------------------------------------
# SimHash
# ---------------------------------------------------------------------------
def simhash_sigs(x, planes, chunk=CHUNK):
    W = planes.shape[0] // 64
    out = np.empty((x.shape[0], W), np.uint64)
    for s in range(0, x.shape[0], chunk):
        bits = (x[s:s + chunk] @ planes.T) >= 0
        out[s:s + chunk] = np.packbits(bits, axis=1, bitorder="little").view("<u8")
    return out


@njit(cache=True)
def _pop(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


@njit(parallel=True, cache=True)
def _ham_counts(dsig, qsig, hstar):
    nq, W = qsig.shape
    out = np.zeros(nq, np.int64)
    for q in prange(nq):
        c = 0
        for i in range(dsig.shape[0]):
            h = 0
            for w in range(W):
                h += _pop(dsig[i, w] ^ qsig[q, w])
            if h <= hstar[q]:
                c += 1
        out[q] = c
    return out


def kmin_simhash(dv, qv, tops, bits, seed=0):
    planes = np.random.default_rng(seed).standard_normal((bits, dv.shape[1])).astype(np.float32)
    dsig, qsig = simhash_sigs(dv, planes), simhash_sigs(qv, planes)
    hstar = np.empty(qv.shape[0], np.int64)
    for q in range(qv.shape[0]):
        hstar[q] = int(np.bitwise_count(dsig[tops[q]] ^ qsig[q]).sum(1).max())
    return _ham_counts(dsig, qsig, hstar)


# ---------------------------------------------------------------------------
# RaBitQ:分块批量距离
# ---------------------------------------------------------------------------
def batch_dist(s, qblks, rows, full):
    """rows 这些文档对全部查询的距离,形状 (len(rows), nq)。full=False -> 第一阶段估计。"""
    d, eb = s["d"], s["ex_bits"]
    signs = s["signs"][rows]
    if not full or eb == 0:
        bits = np.unpackbits(signs.view(np.uint8), axis=1, bitorder="little")[:, :d].astype(np.float32)
        QQ = np.stack([b["qq"] for b in qblks], 1).astype(np.float32)          # d x nq,整数精确
        dot = bits @ QQ
        pc = bits.sum(1, keepdims=True)
        c1 = np.array([b["c1"] for b in qblks], np.float32)
        c2 = np.array([b["c2"] for b in qblks], np.float32)
        c34 = np.array([b["c34"] for b in qblks], np.float32)
        qrc = np.array([b["qr_to_c"] for b in qblks], np.float32)
        fac = s["fac1"][rows]
        final = c1 * dot + c2 * pc - c34
        pre = (fac[:, :1] + qrc) - (np.float32(2) * fac[:, 1:2]) * final
        return np.maximum(np.float32(0), pre)
    vals = R.code_values(signs, s["excode"][rows], d, eb).astype(np.float32)
    RQ = np.stack([b["rq"] for b in qblks], 1).astype(np.float32)              # d x nq
    cb = np.float32(qblks[0]["cb"])
    ex_ip = vals @ RQ + cb * RQ.sum(0)
    qrc = np.array([b["qr_to_c"] for b in qblks], np.float32)
    fx = s["facx"][rows]
    return np.maximum(np.float32(0), qrc + fx[:, :1] + fx[:, 1:2] * ex_ip)


def self_check(s, qblks, full):
    """第一块的前两个查询,与 rabitq_ref 的逐查询实现比 —— 批量算法与参考模型必须一致。"""
    rows = np.arange(min(CHUNK, s["N"]))
    got = batch_dist(s, qblks[:2], rows, full)
    sub = {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim == 2 else v) for k, v in s.items()}
    for j in range(2):
        if full and s["ex_bits"]:
            want = R.stage2_full(sub["signs"], sub["excode"], sub["facx"], qblks[j], s["d"])
        else:
            want = R.stage1_est(sub["signs"], sub["fac1"], qblks[j])
        err = np.abs(got[:, j].astype(np.float64) - want) / np.maximum(np.abs(want), 1e-6)
        if err.max() > 1e-4:
            sys.exit("!! 批量距离与参考模型不一致(nb=%d, 最大相对误差 %.2e)" % (s["nb"], err.max()))


def build_rabitq(dv, nb):
    t0 = time.time()
    idx = faiss.IndexRaBitQ(dv.shape[1], faiss.METRIC_L2, nb)
    idx.train(dv)
    idx.add(dv)
    s = R.split_codes(idx)
    del idx
    return s, time.time() - t0


def kmin_rabitq(s, qblks, tops):
    full = s["ex_bits"] > 0
    self_check(s, qblks, full)
    nq = len(qblks)
    uniq, inv = np.unique(np.concatenate(tops), return_inverse=True)
    dtop = batch_dist(s, qblks, uniq, full)                     # len(uniq) x nq
    inv = inv.reshape(nq, K)
    thr = np.array([dtop[inv[q], q].max() for q in range(nq)], np.float32)
    counts = np.zeros(nq, np.int64)
    for st in range(0, s["N"], CHUNK):
        rows = np.arange(st, min(st + CHUNK, s["N"]))
        counts += (batch_dist(s, qblks, rows, full) <= thr).sum(0)
    return counts


def refine_replay(s, qblks, exact_tops, batch=16):
    """两阶段重放:返回 {tasks: (精算数数组, 召回数组)}。"""
    N = s["N"]
    out = {1: ([], []), 2816: ([], []), "G2816": ([], [])}
    for b0 in range(0, len(qblks), batch):
        qb = qblks[b0:b0 + batch]
        est = np.empty((N, len(qb)), np.float32)
        full = np.empty((N, len(qb)), np.float32)
        for st in range(0, N, CHUNK):
            rows = np.arange(st, min(st + CHUNK, N))
            est[rows] = batch_dist(s, qb, rows, False)
            full[rows] = batch_dist(s, qb, rows, True)
        for j, blk in enumerate(qb):
            for T in (1, 2816, "G2816"):
                if T == "G2816":      # 设备上该用的:全局 τ 两遍(rabitq_ref.two_stage_global)
                    _, _, refined, (_, mi), _ = R.two_stage_global(est[:, j], full[:, j], s["fac1"][:, 2],
                                                                  blk["g_error"], K, 2816)
                else:                 # FAISS 单线程顺序扫描(1),与每 task 各自的堆(2816)
                    _, _, refined, (_, mi) = R.two_stage(est[:, j], full[:, j], s["fac1"][:, 2],
                                                         blk["g_error"], K, T)
                out[T][0].append(int(refined.sum()))
                out[T][1].append(len(set(mi.tolist()) & set(exact_tops[b0 + j].tolist())) / K)
    return {T: (np.array(a), np.array(b)) for T, (a, b) in out.items()}


def summary(kmin, N):
    srt = np.sort(kmin)
    K95 = int(srt[min(len(srt) - 1, int(np.ceil(SUCCESS * len(srt))) - 1)])
    return {"p50": float(np.percentile(kmin, 50)), "p90": float(np.percentile(kmin, 90)),
            "fixed_K95": K95, "share": K95 / N}


def run_size(docs, queries, n, a):
    ids = vecdata.subset_ids(docs.shape[0], n, a.subset_seed)
    dv = vecdata.take(docs, ids)
    N, d = dv.shape
    qv = queries[vecdata.query_ids(queries.shape[0], a.nq, a.query_seed)]
    print("\n==== N = %d (%d 个查询) ====" % (N, qv.shape[0]), flush=True)
    res = {"N": N, "nq": int(qv.shape[0]), "codes": {}, "refine": {}}

    t0 = time.time()
    flat = faiss.IndexFlatIP(d)
    flat.add(dv)
    _, tops = flat.search(qv, K)
    del flat
    tops = [t for t in tops]
    print("  精确 top-%d: %.0f s" % (K, time.time() - t0), flush=True)

    print("  %-14s %6s | %8s %8s %10s %8s" % ("code", "B/vec", "p50", "p90", "K@95%", "share"))
    for bits in (384, 768):
        km = kmin_simhash(dv, qv, tops, bits)
        sm = summary(km, N)
        res["codes"]["SimHash %d" % bits] = {"bytes": bits // 8, **sm}
        print("  %-14s %6d | %8.0f %8.0f %10d %7.3f%%"
              % ("SimHash %d" % bits, bits // 8, sm["p50"], sm["p90"], sm["fixed_K95"], 100 * sm["share"]),
              flush=True)
    for nb in (1, 2, 4):
        s, tb = build_rabitq(dv, nb)
        qblks = [R.query_block(q, s["center"], d, s["W"], qb=4, ex_bits=s["ex_bits"]) for q in qv]
        km = kmin_rabitq(s, qblks, tops)
        sm = summary(km, N)
        nbytes = s["W"] * 8 + (8 if nb == 1 else 12) + s["excode"].shape[1] + (0 if nb == 1 else 8)
        res["codes"]["RaBitQ %d-bit" % nb] = {"bytes": nbytes, "build_s": tb, **sm}
        print("  %-14s %6d | %8.0f %8.0f %10d %7.3f%%   (编码 %.0f s)"
              % ("RaBitQ %d-bit" % nb, nbytes, sm["p50"], sm["p90"], sm["fixed_K95"],
                 100 * sm["share"], tb), flush=True)
        if nb >= 2 and a.nq_refine > 0:
            nr = min(a.nq_refine, len(qblks))
            rr = refine_replay(s, qblks[:nr], tops[:nr])
            for T, (refined, rec) in rr.items():
                tag = "%d-bit_%s" % (nb, T if isinstance(T, str) else "T%d" % T)
                res["refine"][tag] = {
                    "mean": float(refined.mean()), "p90": float(np.percentile(refined, 90)),
                    "share": float(refined.mean() / N), "recall10": float(rec.mean()),
                    "stage2_over_stage1": float(refined.mean() * (s["excode"].shape[1] + 8)
                                                / (N * (s["W"] * 8 + 12)))}
                label = {1: "1 task(FAISS)", 2816: "2816 task 各自堆", "G2816": "2816 task 全局τ"}[T]
                print("    精算 %d-bit, %-16s: 平均 %.0f (p90 %.0f, %.4f%% of N), "
                      "召回@10 %.4f, 第二阶段/第一阶段字节 %.2f%%"
                      % (nb, label, refined.mean(), np.percentile(refined, 90), 100 * refined.mean() / N,
                         rec.mean(), 100 * res["refine"][tag]["stage2_over_stage1"]), flush=True)
        del s
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True, help="encode_msmarco.py 的输出目录,或 .npz")
    ap.add_argument("--sizes", default="100000,1000000,0")
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--nq-refine", type=int, default=100)
    ap.add_argument("--subset-seed", type=int, default=0)
    ap.add_argument("--query-seed", type=int, default=1)
    ap.add_argument("--out", default="scale_results.json")
    a = ap.parse_args()
    docs, queries = vecdata.load(a.emb)
    results = json.load(open(a.out)) if os.path.isfile(a.out) else {}
    for n in [int(x) for x in a.sizes.split(",")]:
        r = run_size(docs, queries, n, a)
        results[str(r["N"])] = r
        json.dump(results, open(a.out, "w"), indent=1)
    print("\nsaved", a.out)


if __name__ == "__main__":
    main()
