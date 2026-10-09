"""开发机上校验 rabitq_ref.py 与 FAISS 1.15.1 一致 —— 没通过就不许写 kernel。

四项检查,每项都对 nb = 1, 2, 4 跑:
  1. 第一阶段:参考模型 1-bit 估计 vs FAISS 1-bit 索引的距离(全部 N 条)
  2. 多 bit 码里的第一阶段因子 == 同数据 1-bit 索引的因子(否则第一阶段不是同一个数)
  3. 第二阶段:参考模型完整距离 vs FAISS 多 bit 索引 k=N 搜索的距离
     (k=N 时堆永远不满、阈值恒为 inf,FAISS 会对每条都精算)
  4. 两阶段 top-k:参考模型(1 个 task 与 2816 个 task)vs FAISS 真实 k=10 搜索

数据:~/.cache/ragdwm/emb 里缓存的 BEIR 嵌入(bge-small, 384 维)。
用法:python test_ref_vs_faiss.py [scifact|nfcorpus|arguana|fiqa] [查询数]
"""
import glob
import os
import sys

import faiss
import numpy as np

import rabitq_ref as R

K = 10


def load(name):
    hits = glob.glob(os.path.join(os.path.expanduser("~"), ".cache", "ragdwm", "emb",
                                  name + "__bge-small__*.npz"))
    if not hits:
        sys.exit("没有 %s 的嵌入缓存,先在 RAGDWM 下跑 report_adaptive_kprime.py %s" % (name, name))
    z = np.load(hits[0])
    return np.ascontiguousarray(z["d"], np.float32), np.ascontiguousarray(z["q"], np.float32)


def by_id(D, I, N):
    out = np.empty(N, np.float32)
    out[I] = D
    return out


def rel_err(a, b):
    scale = max(1e-6, float(np.abs(b).max()))
    return float(np.abs(a.astype(np.float64) - b).max() / scale)


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    nq = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    dv, qv = load(name)
    qv = qv[:nq]
    N, d = dv.shape
    print("%s: N=%d d=%d, %d queries" % (name, N, d, nq))
    i1 = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 1)
    i1.train(dv)
    i1.add(dv)
    s1 = R.split_codes(i1)
    ok = True
    for nb in (1, 2, 4):
        im = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb)
        im.train(dv)
        im.add(dv)
        qb = im.qb
        s = R.split_codes(im)
        # 检查 2:符号码与第一阶段因子相同
        same_signs = bool(np.array_equal(s["signs"], s1["signs"]))
        same_fac = float(np.abs(s["fac1"][:, :2] - s1["fac1"][:, :2]).max())
        e1_max = e2_max = 0.0
        agree1 = agreeT = 0
        for qi in range(nq):
            q = qv[qi]
            qblk = R.query_block(q, s["center"], d, s["W"], qb=qb, ex_bits=s["ex_bits"])
            est = R.stage1_est(s["signs"], s["fac1"], qblk)
            # 检查 1:与 FAISS 1-bit 索引的距离比(同中心、同符号码)
            D, I = i1.search(q[None], N)
            e1_max = max(e1_max, rel_err(est, by_id(D[0], I[0], N)))
            if s["ex_bits"]:
                full = R.stage2_full(s["signs"], s["excode"], s["facx"], qblk, d)
                D, I = im.search(q[None], N)
                e2_max = max(e2_max, rel_err(full, by_id(D[0], I[0], N)))
            else:
                full = est
            # 检查 4:两阶段 top-k
            _, I10 = im.search(q[None], K)
            want = set(I10[0].tolist())
            for T, counter in ((1, "a1"), (2816, "aT")):
                _, _, _, (md, mi) = R.two_stage(est, full, s["fac1"][:, 2], qblk["g_error"], K, T)
                if set(mi.tolist()) == want:
                    if T == 1:
                        agree1 += 1
                    else:
                        agreeT += 1
        line = ("  nb=%d qb=%d | signs==1bit %s, fac diff %.1e | stage1 rel err %.1e | "
                "stage2 rel err %s | top-%d == FAISS: 1 task %d/%d, 2816 tasks %d/%d"
                % (nb, qb, same_signs, same_fac, e1_max,
                   "%.1e" % e2_max if s["ex_bits"] else "  n/a  ", K, agree1, nq, agreeT, nq))
        print(line)
        ok &= same_signs and same_fac == 0.0 and e1_max < 1e-4 and (e2_max < 1e-4) \
            and agree1 >= nq - 1
    print("\n结论:", "参考模型与 FAISS 一致" if ok else "!! 有不一致,先查清再写 kernel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
