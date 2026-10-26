"""faiss_host.py —— 公开库基线:FAISS IndexRaBitQ 在主机上搜同一份码。

host_rabitq.c 是「同一算法的认真实现」;这里是「别人会直接拿来用的库」。两者都要报,
审稿人才不会说基线是我们自己写弱的。

不重新编码:从 SoA 文件拼回 FAISS 的每向量连续布局(AoS),add_sa_codes 直接装进索引
(开发机上核对过:这样得到的索引与 train + add 的搜索结果完全一致)。中心和旋转矩阵来自
encode.py 的 PREFIX.center.npy / PREFIX.rot.npy,查询按同样方式旋转。

FAISS 的平铺 RaBitQ 搜索在查询之间并行(OpenMP),单个查询内部是单线程顺序扫描。
所以报两个数:--batch 1(单查询延迟,只用一个核)和 --batch 全部(吞吐)。

用法:python faiss_host.py --data PREFIX --queries query.fbin --nq 1000 --batch 1000 --dump f.bin
"""
import argparse
import json
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vecio  # noqa: E402


def load_index(prefix):
    meta = np.fromfile(prefix + ".meta", "<u8", 8)
    N, d, W, nb, eb, E = (int(x) for x in meta[1:7])
    sb = (d + 7) // 8
    idx = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb)
    faiss.copy_array_to_vector(np.load(prefix + ".center.npy").astype(np.float32), idx.center)
    idx.is_trained = True
    cs = idx.sa_code_size()
    signs = np.memmap(prefix + ".signs", "<u8", "r", shape=(N, W))
    fac1 = np.memmap(prefix + ".fac1", "<f4", "r", shape=(N, 3))
    ex = np.memmap(prefix + ".excode", np.uint8, "r", shape=(N, E)) if E else None
    facx = np.memmap(prefix + ".facx", "<f4", "r", shape=(N, 2))
    step = 1 << 22
    for s in range(0, N, step):
        e = min(N, s + step)
        n = e - s
        c = np.empty((n, cs), np.uint8)
        c[:, :sb] = np.ascontiguousarray(signs[s:e]).view(np.uint8).reshape(n, W * 8)[:, :sb]
        if eb == 0:
            c[:, sb:sb + 8] = np.ascontiguousarray(fac1[s:e, :2]).view(np.uint8).reshape(n, 8)
        else:
            c[:, sb:sb + 12] = np.ascontiguousarray(fac1[s:e]).view(np.uint8).reshape(n, 12)
            c[:, sb + 12:sb + 12 + E] = ex[s:e]
            c[:, sb + 12 + E:sb + 20 + E] = np.ascontiguousarray(facx[s:e]).view(np.uint8).reshape(n, 8)
        idx.add_sa_codes(c)
    return idx, N, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qkey", default="test")
    ap.add_argument("--column", default="emb")
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0, help="每次 search 的查询数,0 = 全部一次")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--qb", type=int, default=4)
    ap.add_argument("--dump")
    ap.add_argument("--json")
    a = ap.parse_args()
    t0 = time.time()
    idx, N, d = load_index(a.data)
    idx.qb = a.qb
    print("FAISS IndexRaBitQ: N=%d d=%d 载入 %.0f s, 线程 %d" % (N, d, time.time() - t0,
                                                             faiss.omp_get_max_threads()), flush=True)
    qv = np.ascontiguousarray(vecio.load(a.queries, a.nq, key=a.qkey, column=a.column), np.float32)
    rot_path = a.data + ".rot.npy"
    if os.path.isfile(rot_path):
        qv = np.ascontiguousarray(qv @ np.load(rot_path))
    B = a.batch or qv.shape[0]
    idx.search(qv[:1], a.k)                                   # 预热
    D = np.empty((qv.shape[0], a.k), np.float32)
    I = np.empty((qv.shape[0], a.k), np.int64)
    lat = []
    t_all = time.time()
    for s in range(0, qv.shape[0], B):
        t = time.time()
        D[s:s + B], I[s:s + B] = idx.search(qv[s:s + B], a.k)
        lat.append((time.time() - t) * 1e3)
    tot = time.time() - t_all
    out = {"data": a.data, "N": N, "nq": int(qv.shape[0]), "batch": B,
           "batch_ms_median": float(np.median(lat)), "qps": qv.shape[0] / tot}
    print("  每批 %d 个查询:中位 %.2f ms;吞吐 %.1f 查询/s" % (B, out["batch_ms_median"], out["qps"]))
    if a.dump:
        with open(a.dump, "wb") as f:
            for i in range(qv.shape[0]):
                for j in range(a.k):
                    f.write(np.float32(D[i, j]).tobytes())
                    f.write(np.uint32(I[i, j] if I[i, j] >= 0 else 0xFFFFFFFF).tobytes())
                f.write(np.uint64(0).tobytes())          # FAISS 不报精算次数
                f.write(np.uint64(N).tobytes())
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
