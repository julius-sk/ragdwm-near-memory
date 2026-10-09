"""vecdata.py —— 规模验证与 prep_rabitq.py 共用(不叫 datasets:会与 HuggingFace 的 datasets 包重名)的向量加载与子集抽样。

两种来源:
  *.npz              含 d(文档向量)与 q(查询向量),开发机 BEIR 实验的缓存格式
  目录               含 docs.npy 与 queries.npy(scale/encode_msmarco.py 的输出),
                     docs.npy 以 mmap 打开 —— 884 万 x 384 float32 = 13.6 GB,不必整读

子集抽样(subset)必须在 scale_kprime.py 与 prep_rabitq.py 之间一致,否则设备上跑的
和规模验证里统计的不是同一批文档。所以只在这里实现一次。
"""
import os

import numpy as np


def load(path):
    """返回 (docs, queries),docs 可能是 memmap。"""
    if os.path.isdir(path):
        docs = np.load(os.path.join(path, "docs.npy"), mmap_mode="r")
        queries = np.load(os.path.join(path, "queries.npy"))
        return docs, np.ascontiguousarray(queries, np.float32)
    z = np.load(path)
    return np.ascontiguousarray(z["d"], np.float32), np.ascontiguousarray(z["q"], np.float32)


def subset_ids(n_total, n, seed=0):
    """确定性的无放回抽样,结果升序(保持原顺序,mmap 读取也更连续)。n >= n_total 时取全部。"""
    if n <= 0 or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def query_ids(n_total, nq, seed=1):
    """查询抽样;与文档抽样用不同的种子,两者互不影响。"""
    if nq <= 0 or nq >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, nq, replace=False))


def take(docs, ids, chunk=1 << 20):
    """按 ids 从(可能是 memmap 的)docs 取行,分块读,避免一次性物化整个 memmap。"""
    out = np.empty((ids.shape[0], docs.shape[1]), np.float32)
    for s in range(0, ids.shape[0], chunk):
        out[s:s + chunk] = docs[ids[s:s + chunk]]
    return out
