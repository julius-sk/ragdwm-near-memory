"""prep_rabitq.py —— 用 FAISS 编码向量,写出 rabitq_scan 要读的二进制文件。

FAISS 的 IndexRaBitQ 负责编码(RaBitQ 的编码比搜索复杂得多,不值得在设备上重写);
rabitq_ref.split_codes 把 FAISS 每条向量连续存放的码拆成结构体数组(SoA),
rabitq_ref.query_block 生成每个查询的参数块。两者都已对 FAISS 校验。

输出(PREFIX 为 --out):
  PREFIX.meta      8 x uint64: magic 'RBQ1', N, d, words, nb, exBits, exBytes, 0
  PREFIX.signs     N x words x 8 B
  PREFIX.fac1      N x 3 float32   [or_minus_c_l2sqr, dp_multiplier, f_error]
  PREFIX.excode    N x exBytes B   (nb=1 时为空文件)
  PREFIX.facx      N x 2 float32   [f_add_ex, f_rescale_ex]
  PREFIX.qblocks   头 4 x uint64 [magic, nq, qblockBytes, k],后接 nq 个参数块
  PREFIX.check.npz FAISS 的 top-k 与精确 top-k,供 verify_rabitq.py 报告召回

参数块(qblockBytes 字节,64 B 对齐):
  QHeader 64 B = 8 x uint32 [d, words, qb, exBits, exBytes, k, 0, 0]
               + 8 x float  [c1, c2, c34, qrToC, gError, cb, 0, 0]
  planes  qb x words uint64
  rq      d float32
必须与 mu_kernel/mu_rabitq.cpp 的 struct QHeader 完全一致。

用法:
  python prep_rabitq.py --emb fiqa.npz --nb 4 --nq 100 --out data/fiqa_4b
  python prep_rabitq.py --synthetic 1000000 384 --nb 2 --nq 50 --out data/syn1m_2b
"""
import argparse
import os
import struct

import faiss
import numpy as np

import rabitq_ref as R

MAGIC_META = 0x31514252          # 'RBQ1'
MAGIC_QB = 0x31425152            # 'RQB1'


def qblock_bytes(d, words, qb):
    raw = 64 + qb * words * 8 + d * 4
    return (raw + 63) // 64 * 64


def pack_qblock(qblk, d, words, ex_bits, ex_bytes, k, nbytes):
    h = struct.pack("<8I8f", d, words, qblk["qb"], ex_bits, ex_bytes, k, 0, 0,
                    float(qblk["c1"]), float(qblk["c2"]), float(qblk["c34"]),
                    float(qblk["qr_to_c"]), float(qblk["g_error"]), float(qblk["cb"]), 0.0, 0.0)
    body = h + qblk["planes"].astype("<u8").tobytes() + qblk["rq"].astype("<f4").tobytes()
    assert len(body) <= nbytes
    return body + b"\0" * (nbytes - len(body))


def load_vectors(a):
    if a.emb:
        z = np.load(a.emb)
        return np.ascontiguousarray(z["d"], np.float32), np.ascontiguousarray(z["q"], np.float32)
    n, d = a.synthetic
    rng = np.random.default_rng(a.seed)
    dv = rng.standard_normal((n, d), dtype=np.float32)
    dv /= np.linalg.norm(dv, axis=1, keepdims=True)
    pick = rng.choice(n, a.nq, replace=False)
    qv = dv[pick] + 0.3 * rng.standard_normal((a.nq, d), dtype=np.float32) / np.sqrt(d)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)
    return dv, qv.astype(np.float32)


def random_codes(a):
    """只为测带宽:直接生成随机码与随机因子,不经过 FAISS、不生成浮点向量。
    1 亿条 x 384 维的浮点向量要 154 GB,超过 solab-h1 的 128 GB;随机码只要
    N x (60 + 第二阶段字节)。码不对应任何向量,所以输出没有正确性可言,
    第二阶段的精算比例也没有意义 —— 只能用于 E 模式(第一阶段扫描带宽)。"""
    N, d = a.random_codes
    rng = np.random.default_rng(a.seed)
    W = ((d + 7) // 8 + 7) // 8
    eb = a.nb - 1
    E = (d * eb + 7) // 8
    qb = a.qb or 4
    print("随机码:N=%d d=%d nb=%d —— 仅用于带宽测试,不可校验" % (N, d, a.nb))
    with open(a.out + ".meta", "wb") as f:
        f.write(struct.pack("<8Q", MAGIC_META, N, d, W, a.nb, eb, E, 0))
    chunk = 1 << 22
    with open(a.out + ".signs", "wb") as fs, open(a.out + ".fac1", "wb") as f1, \
            open(a.out + ".excode", "wb") as fe, open(a.out + ".facx", "wb") as fx:
        for s in range(0, N, chunk):
            n = min(chunk, N - s)
            rng.integers(0, 2 ** 64, (n, W), dtype=np.uint64).astype("<u8").tofile(fs)
            fac = np.stack([rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.0, n),
                            rng.uniform(0.0, 0.1, n)], 1).astype("<f4")
            fac.tofile(f1)
            if E:
                rng.integers(0, 256, (n, E), dtype=np.uint8).tofile(fe)
            np.stack([rng.uniform(-1, 1, n), rng.uniform(0.5, 1.5, n)], 1).astype("<f4").tofile(fx)
    center = np.zeros(d, np.float32)
    qbb = qblock_bytes(d, W, qb)
    with open(a.out + ".qblocks", "wb") as f:
        f.write(struct.pack("<4Q", MAGIC_QB, a.nq, qbb, a.k))
        for _ in range(a.nq):
            q = rng.standard_normal(d).astype(np.float32)
            q /= np.linalg.norm(q)
            f.write(pack_qblock(R.query_block(q, center, d, W, qb=qb, ex_bits=eb), d, W, eb, E, a.k, qbb))
    print("写出 %s.*(随机码)" % a.out)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--emb", help="npz,含 d(文档向量)和 q(查询向量)")
    g.add_argument("--synthetic", nargs=2, type=int, metavar=("N", "D"))
    g.add_argument("--random-codes", nargs=2, type=int, metavar=("N", "D"),
                   help="只测带宽:随机码,不经过 FAISS")
    ap.add_argument("--nb", type=int, default=4, help="每维位数 1..8")
    ap.add_argument("--qb", type=int, default=0, help="查询量化位数,0 = 用 FAISS 默认")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.random_codes:
        random_codes(a)
        return

    dv, qv = load_vectors(a)
    qv = qv[:a.nq]
    N, d = dv.shape
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    idx = faiss.IndexRaBitQ(d, faiss.METRIC_L2, a.nb)
    if a.qb:
        idx.qb = a.qb
    idx.train(dv)
    idx.add(dv)
    s = R.split_codes(idx)
    W, eb = s["W"], s["ex_bits"]
    E = s["excode"].shape[1]
    print("N=%d d=%d nb=%d qb=%d | 第一阶段 %d B/向量, 第二阶段 %d B/向量"
          % (N, d, a.nb, idx.qb, W * 8 + 12, E + 8))

    with open(a.out + ".meta", "wb") as f:
        f.write(struct.pack("<8Q", MAGIC_META, N, d, W, a.nb, eb, E, 0))
    s["signs"].astype("<u8").tofile(a.out + ".signs")
    s["fac1"].astype("<f4").tofile(a.out + ".fac1")
    s["excode"].tofile(a.out + ".excode")
    s["facx"].astype("<f4").tofile(a.out + ".facx")

    qbb = qblock_bytes(d, W, idx.qb)
    with open(a.out + ".qblocks", "wb") as f:
        f.write(struct.pack("<4Q", MAGIC_QB, qv.shape[0], qbb, a.k))
        for q in qv:
            qblk = R.query_block(q, s["center"], d, W, qb=idx.qb, ex_bits=eb)
            f.write(pack_qblock(qblk, d, W, eb, E, a.k, qbb))

    # 召回参照:FAISS 自己的 top-k、精确 L2 top-k
    _, faiss_top = idx.search(qv, a.k)
    flat = faiss.IndexFlatL2(d)
    flat.add(dv)
    _, exact_top = flat.search(qv, a.k)
    np.savez(a.out + ".check.npz", faiss_top=faiss_top, exact_top=exact_top, center=s["center"])
    print("写出 %s.{meta,signs,fac1,excode,facx,qblocks,check.npz}, %d 个查询, 参数块 %d B"
          % (a.out, qv.shape[0], qbb))


if __name__ == "__main__":
    main()
