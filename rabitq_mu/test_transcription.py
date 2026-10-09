"""把 mu_kernel/mu_rabitq.cpp 逐句转写(transcribe)成 Python,与 rabitq_ref.py 比对。

开发机没有 C++ 编译器,kernel 是盲写的。这里每个函数都按 C++ 的语句顺序、
循环结构和 float32 精度一句一句照抄,用来抓 C++ 里的**逻辑**错误(下标、移位、
分块、堆替换规则)。它抓不到编译器层面的问题(-O3 误编译、FMA 合并造成的舍入
差异)—— 那些只能在 solab-h1 上用 I/E/F 三种诊断模式分层排查。

改了 mu_rabitq.cpp 就必须同步改这里,否则这份测试失去意义。

用法:python test_transcription.py [scifact] [向量数] [查询数]
"""
import sys

import faiss
import numpy as np

import rabitq_ref as R
from test_ref_vs_faiss import load

f32 = np.float32
NONE = 0xFFFFFFFF
INF = f32(np.inf)


# ---- 转写:匿名命名空间里的辅助函数 -------------------------------------------
def popcount64(x):
    return bin(int(x)).count("1")


def myRange(n, taskIdx, taskCount):
    perTask = (n + taskCount - 1) // taskCount
    begin = taskIdx * perTask
    end = begin + perTask
    if end > n:
        end = n
    if begin > n:
        begin = n
    return begin, end


def stage1Ints(s, q):
    dot = 0
    pc = 0
    for w in range(q["words"]):
        x = int(s[w])
        pc += popcount64(x)
        for j in range(q["qb"]):
            dot += popcount64(int(q["planes"][j * q["words"] + w]) & x) << j
    return dot, pc


def stage1Est(dot, pc, f1, h):
    fd = f32(0.0)
    fd = f32(fd + f32(h["c1"] * f32(dot)))
    fd = f32(fd + f32(h["c2"] * f32(pc)))
    fd = f32(fd - h["c34"])
    pre = f32(f32(f1[0] + h["qrToC"]) - f32(f32(f32(2.0) * f1[1]) * fd))
    return pre if pre > f32(0.0) else f32(0.0)


def stage2Full(s, ex, fx, q):
    eb = q["exBits"]
    mask = (1 << eb) - 1
    acc = f32(0.0)
    for i in range(q["d"]):
        sb = (int(s[i >> 6]) >> (i & 63)) & 1
        bp = i * eb
        byteIdx = bp >> 3
        off = bp & 7
        raw = int(ex[byteIdx])
        if off + eb > 8:
            raw |= int(ex[byteIdx + 1]) << 8
        exv = (raw >> off) & mask
        acc = f32(acc + f32(q["rq"][i] * f32(f32((sb << eb) + exv) + q["cb"])))
    dist = f32(f32(q["qrToC"] + fx[0]) + f32(fx[1] * acc))
    return dist if dist > f32(0.0) else f32(0.0)


def rabitq_topk_task(signs, fac1, excode, facx, q, numSigs, taskIdx, taskCount):
    k = q["k"]
    hd = [INF] * k
    hi = [NONE] * k
    begin, end = myRange(numSigs, taskIdx, taskCount)
    refined = 0
    for i in range(begin, end):
        thr = hd[0]
        jmax = 0
        for j in range(1, k):
            if hd[j] > thr:
                thr = hd[j]
                jmax = j
        dot, pc = stage1Ints(signs[i], q)
        est = stage1Est(dot, pc, fac1[i], q)
        lb = f32(est - f32(fac1[i][2] * q["gError"]))
        if lb < f32(0.0):
            lb = f32(0.0)
        if lb < thr:
            refined += 1
            full = est if q["exBits"] == 0 else stage2Full(signs[i], excode[i], facx[i], q)
            if full < thr:
                hd[jmax] = full
                hi[jmax] = i
    return hd, hi, refined


def rabitq_merge(res_d, res_i, numTasks, k):
    """转写 rabitq_merge:(距离, id) 字典序维护最好的 k 个。"""
    bd = [INF] * k
    bi = [NONE] * k
    for t in range(numTasks):
        for j in range(k):
            id_ = int(res_i[t][j])
            if id_ == NONE:
                continue
            dv = res_d[t][j]
            jw = 0
            for jj in range(1, k):
                if bd[jj] > bd[jw] or (bd[jj] == bd[jw] and bi[jj] > bi[jw]):
                    jw = jj
            if dv < bd[jw] or (dv == bd[jw] and id_ < bi[jw]):
                bd[jw] = dv
                bi[jw] = id_
    return bd, bi


def as_query(qblk, s, k):
    return {"d": s["d"], "words": s["W"], "qb": qblk["qb"], "exBits": s["ex_bits"],
            "k": k, "c1": qblk["c1"], "c2": qblk["c2"], "c34": qblk["c34"],
            "qrToC": qblk["qr_to_c"], "gError": qblk["g_error"], "cb": qblk["cb"],
            "planes": qblk["planes"].ravel(), "rq": qblk["rq"]}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "scifact"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    nq = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    K, T = 10, 7                                  # 7 个 task:故意不整除,检验 ceil 分块
    dv, qv = load(name)
    dv, qv = dv[:n], qv[:nq]
    d = dv.shape[1]
    bad = 0
    for nb in (1, 2, 4):
        idx = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb)
        idx.train(dv)
        idx.add(dv)
        s = R.split_codes(idx)
        n_int = n_est = n_full = n_task = 0
        e_est = e_full = 0.0
        for qi in range(nq):
            qblk = R.query_block(qv[qi], s["center"], d, s["W"], qb=idx.qb, ex_bits=s["ex_bits"])
            q = as_query(qblk, s, K)
            dot_r, pc_r = R.stage1_ints(s["signs"], qblk["planes"])
            est_r = R.stage1_est(s["signs"], s["fac1"], qblk)
            full_r = (R.stage2_full(s["signs"], s["excode"], s["facx"], qblk, d)
                      if s["ex_bits"] else est_r)
            for i in range(n):
                dot, pc = stage1Ints(s["signs"][i], q)
                n_int += (dot != dot_r[i]) or (pc != pc_r[i])
                est = stage1Est(dot, pc, s["fac1"][i], q)
                e_est = max(e_est, abs(float(est) - float(est_r[i])) / max(1e-6, float(est_r[i])))
                if s["ex_bits"]:
                    full = stage2Full(s["signs"][i], s["excode"][i], s["facx"][i], q)
                    e_full = max(e_full, abs(float(full) - float(full_r[i])) / max(1e-6, float(full_r[i])))
            rd, ri, rref, (md, mi) = R.two_stage(est_r, full_r, s["fac1"][:, 2], qblk["g_error"], K, T)
            td, ti = [], []
            for t in range(T):
                hd, hi, ref = rabitq_topk_task(s["signs"], s["fac1"], s["excode"], s["facx"],
                                               q, n, t, T)
                n_task += (set(hi) != set(ri[t].tolist())) or (ref != rref[t])
                td.append(hd)
                ti.append(hi)
            bd, bi = rabitq_merge(td, ti, T, K)
            n_task += set(b for b in bi if b != NONE) != set(mi.tolist())
        print("  nb=%d: 整数不一致 %d/%d | est 最大相对误差 %.1e | full 最大相对误差 %s | "
              "task 结果不一致 %d/%d"
              % (nb, n_int, n * nq, e_est, "%.1e" % e_full if s["ex_bits"] else "n/a",
                 n_task, T * nq))
        bad += n_int + n_task + (e_est > 1e-5) + (e_full > 1e-5)
    print("\n结论:", "C++ 转写与参考模型一致" if bad == 0 else "!! 转写与参考模型不一致")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
