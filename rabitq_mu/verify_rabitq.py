"""verify_rabitq.py —— 校验 rabitq_scan 的设备输出,与参考模型逐项比对。

读的正是送进设备的那些字节(PREFIX.* 与参数块),不是重新编码 —— 所以
校验的是"设备对这批输入算得对不对",而不是"FAISS 两次编码是否一样"。

  I  整数 (dot, popcount)       必须逐个相等
  E  第一阶段距离               相对误差 < 1e-5(FMA 合并等会造成 ulp 级差异)
  F  第二阶段距离               相对误差 < 1e-4(d 维浮点累加,顺序不同)
  T  top-k                      id 集合与参考模型(同 task 数)一致;
                                另报精算次数差、与 FAISS / 精确 top-k 的召回

用法:
  python verify_rabitq.py --data data/fiqa_4b --mode T --dump out_T.bin --tasks 2816
  python verify_rabitq.py --data data/fiqa_4b --mode E --dump out_E.bin
  python verify_rabitq.py --data data/fiqa_4b --simulate T --dump sim_T.bin --tasks 2816
      (--simulate:按 rabitq_scan 的输出格式用参考模型生成假输出,开发机上检验整条链路)
"""
import argparse
import struct
import sys

import numpy as np

import rabitq_ref as R

MAGIC_META = 0x31514252
MAGIC_QB = 0x31425152


def load_files(prefix):
    meta = np.fromfile(prefix + ".meta", "<u8")
    if meta[0] != MAGIC_META:
        sys.exit("%s.meta 不是 prep_rabitq.py 的输出" % prefix)
    N, d, W, nb, eb, E = (int(x) for x in meta[1:7])
    s = {"N": N, "d": d, "W": W, "nb": nb, "ex_bits": eb, "E": E,
         "signs": np.fromfile(prefix + ".signs", "<u8").reshape(N, W),
         "fac1": np.fromfile(prefix + ".fac1", "<f4").reshape(N, 3),
         "excode": np.fromfile(prefix + ".excode", np.uint8).reshape(N, E),
         "facx": np.fromfile(prefix + ".facx", "<f4").reshape(N, 2)}
    raw = open(prefix + ".qblocks", "rb").read()
    magic, nq, qbb, k = struct.unpack_from("<4Q", raw, 0)
    if magic != MAGIC_QB:
        sys.exit("%s.qblocks 头不对" % prefix)
    qblks = []
    for qi in range(nq):
        b = 32 + qi * qbb
        hdr = struct.unpack_from("<8I8f", raw, b)
        dd, ww, qb, ebb, EE, kk = hdr[:6]
        assert (dd, ww, ebb, EE, kk) == (d, W, eb, E, k), "参数块头与 meta 不一致"
        planes = np.frombuffer(raw, "<u8", qb * W, b + 64).reshape(qb, W)
        rq = np.frombuffer(raw, "<f4", d, b + 64 + qb * W * 8)
        c1, c2, c34, qrc, g, cb = (np.float32(x) for x in hdr[8:14])
        qblks.append({"qb": qb, "ex_bits": eb, "c1": c1, "c2": c2, "c34": c34, "qr_to_c": qrc,
                      "g_error": g, "cb": cb, "planes": planes, "rq": rq})
    return s, qblks, int(k)


def reference(s, qblk):
    est = R.stage1_est(s["signs"], s["fac1"], qblk)
    full = R.stage2_full(s["signs"], s["excode"], s["facx"], qblk, s["d"]) if s["ex_bits"] else est
    return est, full


def simulate(s, qblks, k, mode, dump, tasks, dump_nq, nq):
    """按 rabitq_scan.cpp 的写出格式生成假设备输出。"""
    with open(dump, "wb") as f:
        for qi in range(nq):
            qblk = qblks[qi]
            if mode == "T":
                est, full = reference(s, qblk)
                _, _, refined, (md, mi) = R.two_stage(est, full, s["fac1"][:, 2], qblk["g_error"], k, tasks)
                for dv, iv in zip(md, mi):
                    f.write(struct.pack("<fI", float(dv), int(iv)))
                f.write(struct.pack("<QQ", int(refined.sum()), s["N"]))
            elif qi < dump_nq:
                if mode == "I":
                    dot, pc = R.stage1_ints(s["signs"], qblk["planes"])
                    np.stack([dot, pc], 1).astype("<u4").tofile(f)
                else:
                    est, full = reference(s, qblk)
                    (est if mode == "E" else full).astype("<f4").tofile(f)
    print("模拟输出已写出 ->", dump)


def verify(s, qblks, k, mode, dump, tasks, dump_nq, nq, prefix):
    raw = np.fromfile(dump, np.uint8)
    N = s["N"]
    ok = True
    if mode in "IEF":
        per = N * 8 if mode == "I" else N * 4
        nd = min(dump_nq, raw.size // per)
        if nd == 0:
            sys.exit("dump 里没有完整的查询输出")
        for qi in range(nd):
            chunk = raw[qi * per:(qi + 1) * per]
            qblk = qblks[qi]
            if mode == "I":
                got = chunk.view("<u4").reshape(N, 2).astype(np.int64)
                dot, pc = R.stage1_ints(s["signs"], qblk["planes"])
                bad = int(np.sum((got[:, 0] != dot) | (got[:, 1] != pc)))
                print("  query %d: 整数不一致 %d / %d" % (qi, bad, N))
                if bad:
                    j = int(np.argmax((got[:, 0] != dot) | (got[:, 1] != pc)))
                    print("    首个不一致 i=%d: 设备 (%d,%d) 参考 (%d,%d)"
                          % (j, got[j, 0], got[j, 1], dot[j], pc[j]))
                ok &= bad == 0
            else:
                got = chunk.view("<f4").astype(np.float64)
                est, full = reference(s, qblk)
                want = (est if mode == "E" else full).astype(np.float64)
                err = np.abs(got - want) / np.maximum(np.abs(want), 1e-6)
                tol = 1e-5 if mode == "E" else 1e-4
                j = int(np.argmax(err))
                print("  query %d: 最大相对误差 %.2e (i=%d 设备 %.7g 参考 %.7g), 超容差 %d 个"
                      % (qi, err[j], j, got[j], want[j], int(np.sum(err > tol))))
                ok &= bool(np.all(err <= tol)) and bool(np.all(np.isfinite(got)))
    else:
        rec = k * 8 + 16
        nd = min(nq, raw.size // rec)
        chk = np.load(prefix + ".check.npz")
        agree = refined_diff = bad_q = 0
        rec_exact = rec_faiss = 0.0
        for qi in range(nd):
            r = raw[qi * rec:(qi + 1) * rec]
            pairs = r[:k * 8].view("<u4").reshape(k, 2)
            ids = pairs[:, 1].astype(np.int64)
            dists = pairs[:, 0].copy().view("<f4").astype(np.float64)
            refined, processed = struct.unpack("<QQ", r[k * 8:].tobytes())
            est, full = reference(s, qblks[qi])
            _, _, ref_refined, (md, mi) = R.two_stage(est, full, s["fac1"][:, 2],
                                                      qblks[qi]["g_error"], k, tasks)
            agree += set(ids.tolist()) == set(mi.tolist())
            # 判据不按查询个数容忍,而是逐条核对:
            #  (a) 每个设备 id 在参考模型里的完整距离 == 设备报的距离 —— 编造/错位的 id 过不了
            #  (b) 设备的 k 个距离排序后 == 参考模型的 k 个距离 —— 漏掉更近的过不了
            # 只有 ulp 级舍入造成的打平翻转能同时满足两条。
            valid = (ids >= 0) & (ids < N)
            fa = np.full(k, np.inf)
            fa[valid] = full[ids[valid]]
            ok_a = bool(valid.all()) and bool(np.all(np.abs(fa - dists) <= 1e-4 * np.maximum(np.abs(fa), 1e-6)))
            ok_b = bool(np.allclose(np.sort(dists), np.sort(md.astype(np.float64)), rtol=1e-4, atol=1e-7))
            if not (ok_a and ok_b):
                bad_q += 1
                if bad_q <= 3:
                    print("  !! query %d: 距离核对失败 (id 与距离对得上: %s, 距离集合一致: %s)"
                          % (qi, ok_a, ok_b))
            refined_diff = max(refined_diff, abs(int(refined) - int(ref_refined.sum())))
            if processed != N:
                print("  !! query %d: 处理数 %d != N %d" % (qi, processed, N))
                ok = False
            rec_exact += len(set(ids.tolist()) & set(chk["exact_top"][qi].tolist())) / k
            rec_faiss += len(set(ids.tolist()) & set(chk["faiss_top"][qi].tolist())) / k
        print("  top-%d id 集合与参考模型一致: %d / %d(不一致的必须能由距离打平解释)" % (k, agree, nd))
        print("  距离核对失败的查询: %d / %d" % (bad_q, nd))
        print("  精算次数与参考模型的最大差: %d(浮点 ulp 差异可让个别候选的判定翻转)" % refined_diff)
        print("  召回 vs 精确 L2 top-%d: %.4f | 与 FAISS top-%d 重合: %.4f"
              % (k, rec_exact / nd, k, rec_faiss / nd))
        ok &= bad_q == 0
    print("\n结论:", "设备输出与参考模型一致" if ok else "!! 不一致 —— 按 I -> E -> F -> T 的顺序排查")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=list("IEFT"))
    g.add_argument("--simulate", choices=list("IEFT"))
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tasks", type=int, default=2816)
    ap.add_argument("--dump-nq", type=int, default=2)
    ap.add_argument("--nq", type=int, default=0, help="0 = 全部")
    a = ap.parse_args()
    s, qblks, k = load_files(a.data)
    nq = a.nq or len(qblks)
    if a.simulate:
        simulate(s, qblks, k, a.simulate, a.dump, a.tasks, a.dump_nq, nq)
        return 0
    return verify(s, qblks, k, a.mode, a.dump, a.tasks, a.dump_nq, nq, a.data)


if __name__ == "__main__":
    sys.exit(main())
