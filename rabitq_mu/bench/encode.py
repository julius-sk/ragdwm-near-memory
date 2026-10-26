"""encode.py —— 把公开数据集流式编码成 rabitq_scan 要读的 SoA 文件(与 prep_rabitq.py 同格式)。

prep_rabitq.py 会把全部向量读进内存再交给 FAISS,DEEP1B(384 GB)这样做不行。这里分块:
  第 1 遍  逐块累加,求精确均值作为中心(center)。FAISS IndexRaBitQ.train() 做的也是取均值,
           只是它要全部数据在内存里。
  第 2 遍  每块:可选随机旋转 -> FAISS sa_encode -> 拆成 SoA -> 追加写文件。
已在开发机上核对:先设好 center 再 sa_encode,得到的码与整库 train + add 逐字节相同。

随机旋转(random rotation)。RaBitQ 论文的误差界要求先做随机正交变换;FAISS 的 IndexRaBitQ
本身不做(要用户自己套 IndexPreTransform)。RAGDWM 在 bge 上测过旋转没有影响,但 SIFT、DEEP
这类各维方差差别很大的数据上不能这样假设,所以默认开启(--rotate 0 关闭)。旋转矩阵存进
PREFIX.rot.npy,查询在生成参数块之前用同一个矩阵旋转。旋转是正交变换,L2 距离不变,
所以地面真值(ground truth)不受影响,仍用原始向量计算。

度量:kernel 只实现了 L2。内积数据集只有在向量已归一化时才等价(L2^2 = 2 - 2<x,q>);
Text2Image-1B 这类未归一化的内积数据集不能直接用,脚本会检查并拒绝。

输出:PREFIX.{meta,signs,fac1,excode,facx,qblocks},外加 PREFIX.center.npy、PREFIX.rot.npy、
PREFIX.info.json(数据集、N、参数、各文件字节数)。

用法:
  python encode.py --base base.1B.fbin --crop 100000000 --queries query.public.10K.fbin \
                   --nb 4 --nq 10000 --out /data/rbq/deep100m_4b
"""
import argparse
import json
import os
import shutil
import struct
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import rabitq_ref as R                     # noqa: E402
from prep_rabitq import MAGIC_META, MAGIC_QB, pack_qblock, qblock_bytes   # noqa: E402
import vecio                               # noqa: E402


def split_code_bytes(codes, d, nb):
    """与 rabitq_ref.split_codes 相同的拆法,但作用于一块码字节而不是整个 FAISS 索引。"""
    n = codes.shape[0]
    sb = (d + 7) // 8
    W = (sb + 7) // 8
    sbytes = np.zeros((n, W * 8), np.uint8)
    sbytes[:, :sb] = codes[:, :sb]
    signs = sbytes.view("<u8").reshape(n, W)
    fac1 = np.zeros((n, 3), np.float32)
    ex_bits = nb - 1
    E = (d * ex_bits + 7) // 8
    if ex_bits == 0:
        fac1[:, :2] = codes[:, sb:sb + 8].copy().view("<f4").reshape(n, 2)
        excode = np.zeros((n, 0), np.uint8)
        facx = np.zeros((n, 2), np.float32)
    else:
        fac1[:] = codes[:, sb:sb + 12].copy().view("<f4").reshape(n, 3)
        excode = codes[:, sb + 12:sb + 12 + E].copy()
        facx = codes[:, sb + 12 + E:sb + 12 + E + 8].copy().view("<f4").reshape(n, 2)
    return signs, fac1, excode, facx, W, E


def random_rotation(d, seed):
    """Haar 分布的随机正交矩阵:高斯矩阵做 QR,再按 R 的对角线符号修正。"""
    g = np.random.default_rng(seed).standard_normal((d, d))
    q, r = np.linalg.qr(g)
    q *= np.sign(np.diag(r))
    return q.astype(np.float32)


def write_qblocks(path, qr, center, d, W, qb, nb, E, k):
    qbb = qblock_bytes(d, W, qb)
    with open(path, "wb") as f:
        f.write(struct.pack("<4Q", MAGIC_QB, qr.shape[0], qbb, k))
        for q in qr:
            f.write(pack_qblock(R.query_block(q, center, d, W, qb=qb, ex_bits=nb - 1),
                                d, W, nb - 1, E, k, qbb))


def qblocks_only(a):
    src = a.qblocks_only
    meta = np.fromfile(src + ".meta", "<u8", 8)
    N, d, W, nb, eb, E = (int(x) for x in meta[1:7])
    for ext in ("meta", "signs", "fac1", "excode", "facx", "center.npy", "rot.npy"):
        s, t = src + "." + ext, a.out + "." + ext
        if os.path.exists(s) and not os.path.exists(t):
            try:
                os.symlink(os.path.abspath(s), t)
            except OSError:              # Windows 开发机没有建符号链接的权限:退回复制
                shutil.copyfile(s, t)
    qv = np.ascontiguousarray(vecio.load(a.queries, a.nq, key=a.qkey, column=a.column), np.float32)
    rot = np.load(src + ".rot.npy") if os.path.exists(src + ".rot.npy") else None
    qr = qv @ rot if rot is not None else qv
    write_qblocks(a.out + ".qblocks", qr, np.load(src + ".center.npy"), d, W, a.qb, nb, E, a.k)
    print("写出 %s.qblocks(k=%d,%d 个查询),码文件链接到 %s" % (a.out, a.k, qr.shape[0], src))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="基础向量文件或 parquet 目录")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--crop", type=int, default=0, help="只取前 N 条(big-ann 约定),0 = 全部")
    ap.add_argument("--column", default="emb", help="parquet 列名")
    ap.add_argument("--qkey", default="test", help="hdf5 里查询的键")
    ap.add_argument("--nb", type=int, default=4, help="每维总位数 1..8(1 = 只有符号码)")
    ap.add_argument("--qb", type=int, default=4, help="查询量化位数(FAISS 默认 4)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=0, help="参数块里放几个查询,0 = 全部")
    ap.add_argument("--rotate", type=int, default=1, help="随机旋转的种子;0 = 不旋转")
    ap.add_argument("--chunk", type=int, default=1 << 22)
    ap.add_argument("--normalized-ip", action="store_true",
                    help="数据集声明的度量是内积:检查向量已归一化,之后按 L2 处理")
    ap.add_argument("--qblocks-only", metavar="SRC",
                    help="不重新编码:沿用 SRC 前缀的码、中心和旋转,只按新的 --k 写参数块到 --out"
                         "(--out 下的码文件用符号链接指向 SRC)。用于同一份码分别跑 k=10 与 k'=64")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.qblocks_only:
        return qblocks_only(a)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    base = vecio.load(a.base, a.crop, key="train", column=a.column)
    qv = np.ascontiguousarray(vecio.load(a.queries, a.nq, key=a.qkey, column=a.column), np.float32)
    N, d = base.shape
    print("base %s: N=%d d=%d dtype=%s | queries %d" % (a.base, N, d, base.dtype, qv.shape[0]), flush=True)
    if a.normalized_ip:
        probe = np.asarray(base[:10000], np.float32)
        nrm = np.linalg.norm(probe, axis=1)
        if np.abs(nrm - 1).max() > 1e-3:
            sys.exit("声明为内积,但向量没有归一化(范数 %.3f..%.3f):kernel 只做 L2,不能用"
                     % (nrm.min(), nrm.max()))

    # ---- 第 1 遍:均值 ----
    t0 = time.time()
    acc = np.zeros(d, np.float64)
    for s, x in vecio.chunks(base, a.chunk):
        acc += x.sum(0, dtype=np.float64)
    mean = (acc / N).astype(np.float32)
    rot = random_rotation(d, a.rotate) if a.rotate else None
    center = mean @ rot if rot is not None else mean          # 旋转后的均值 = 均值旋转
    np.save(a.out + ".center.npy", center)
    if rot is not None:
        np.save(a.out + ".rot.npy", rot)
    print("第 1 遍(均值)%.0f s" % (time.time() - t0), flush=True)

    # ---- 第 2 遍:编码 ----
    idx = faiss.IndexRaBitQ(d, faiss.METRIC_L2, a.nb)
    idx.qb = a.qb
    faiss.copy_array_to_vector(center, idx.center)
    idx.is_trained = True
    sb = (d + 7) // 8
    W = (sb + 7) // 8
    E = (d * (a.nb - 1) + 7) // 8
    with open(a.out + ".meta", "wb") as f:
        f.write(struct.pack("<8Q", MAGIC_META, N, d, W, a.nb, a.nb - 1, E, 0))
    t0 = time.time()
    files = {k: open(a.out + "." + k, "wb") for k in ("signs", "fac1", "excode", "facx")}
    try:
        for s, x in vecio.chunks(base, a.chunk):
            if rot is not None:
                x = x @ rot
            codes = idx.sa_encode(x)
            signs, fac1, excode, facx, _, _ = split_code_bytes(codes, d, a.nb)
            signs.astype("<u8").tofile(files["signs"])
            fac1.astype("<f4").tofile(files["fac1"])
            excode.tofile(files["excode"])
            facx.astype("<f4").tofile(files["facx"])
            done = s + x.shape[0]
            el = time.time() - t0
            print("  编码 %d / %d  (%.0f s,预计还要 %.0f s)" % (done, N, el, el / done * (N - done)),
                  flush=True)
    finally:
        for f in files.values():
            f.close()

    # ---- 查询参数块 ----
    qr = qv @ rot if rot is not None else qv
    write_qblocks(a.out + ".qblocks", qr, center, d, W, a.qb, a.nb, E, a.k)
    info = {"base": os.path.abspath(a.base), "queries": os.path.abspath(a.queries), "crop": a.crop,
            "N": int(N), "d": int(d), "nb": a.nb, "qb": a.qb, "k": a.k, "nq": int(qr.shape[0]),
            "rotate": a.rotate, "words": int(W), "exBytes": int(E),
            "bytes_stage1": int(N * (W * 8 + 12)), "bytes_stage2": int(N * (E + 8)),
            "bytes_lb": int(N * 4)}
    with open(a.out + ".info.json", "w") as f:
        json.dump(info, f, indent=1)
    print("写出 %s.*  第一阶段 %.2f GB,第二阶段 %.2f GB,G 模式下界 %.2f GB"
          % (a.out, info["bytes_stage1"] / 1e9, info["bytes_stage2"] / 1e9, info["bytes_lb"] / 1e9))


if __name__ == "__main__":
    main()
