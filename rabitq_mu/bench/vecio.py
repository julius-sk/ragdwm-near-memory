"""vecio.py —— 读公开 ANN 基准的各种向量文件格式,统一成 (n, d) 的 numpy 数组或 memmap。

支持的格式(按后缀判断):
  .fbin / .u8bin / .i8bin / .ibin   big-ann-benchmarks 格式:头 8 B = int32 n, int32 d,后接 n x d 个元素
  .fvecs / .ivecs / .bvecs           TEXMEX 格式(SIFT、GIST):每行前 4 B 是 int32 d
  .npy                               numpy
  .hdf5                              ann-benchmarks 格式:键 train / test / neighbors / distances
  目录(含 *.parquet)                HuggingFace 上的嵌入数据集,取 --column 指定的列

「裁剪(crop)」沿用 big-ann-benchmarks 的约定:10M / 100M 子集 = 基础集的前 N 条,
不是随机抽样。这样我们的子集和别人论文里的 DEEP-10M、DEEP-100M 是同一批向量。

大文件一律 memmap,不整读:DEEP1B 的 base.1B.fbin 有 384 GB,h1 主机内存只有 128 GB。
uint8 / int8 向量(BIGANN、MSSPACEV)读出来时转成 float32,按块转,不整体物化。
"""
import glob
import os

import numpy as np

_BIGANN = {".fbin": np.float32, ".u8bin": np.uint8, ".i8bin": np.int8, ".ibin": np.int32}


def _bigann(path, n_max):
    dt = _BIGANN[os.path.splitext(path)[1]]
    with open(path, "rb") as f:
        n, d = np.fromfile(f, np.int32, 2)
    # 头里写的是全集条数;只下载了前一段(curl -r,用于 10M / 100M 裁剪)时以文件实际长度为准
    n = min(int(n), (os.path.getsize(path) - 8) // (int(d) * np.dtype(dt).itemsize))
    n = n if n_max <= 0 else min(n, n_max)
    return np.memmap(path, dtype=dt, mode="r", offset=8, shape=(n, int(d)))


def _texmex(path, n_max):
    ext = os.path.splitext(path)[1]
    dt = {".fvecs": np.float32, ".ivecs": np.int32, ".bvecs": np.uint8}[ext]
    with open(path, "rb") as f:
        d = int(np.fromfile(f, np.int32, 1)[0])
    item = np.dtype(dt).itemsize
    row = 4 + d * item
    n = os.path.getsize(path) // row
    if n_max > 0:
        n = min(n, n_max)
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(n, row))
    return raw[:, 4:].view(dt).reshape(n, d)        # 跨步视图:每行跳过 4 B 的维度字段


def _parquet(path, column, n_max):
    import pyarrow.parquet as pq
    files = sorted(glob.glob(os.path.join(path, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError("%s 下没有 parquet 文件" % path)
    parts, total = [], 0
    for fn in files:
        col = pq.read_table(fn, columns=[column]).column(column)
        a = np.asarray(col.combine_chunks().flatten()).astype(np.float32)
        d = len(col[0])
        parts.append(a.reshape(-1, d))
        total += parts[-1].shape[0]
        if 0 < n_max <= total:
            break
    out = np.concatenate(parts)
    return out[:n_max] if n_max > 0 else out


def load(path, n_max=0, key=None, column="emb"):
    """返回 (n, d) 数组(大文件是 memmap)。n_max > 0 时只取前 n_max 条(big-ann 的 crop 约定)。
    key:hdf5 的数据集名(train / test / neighbors)。column:parquet 的列名。"""
    if os.path.isdir(path):
        return _parquet(path, column, n_max)
    ext = os.path.splitext(path)[1]
    if ext in _BIGANN:
        return _bigann(path, n_max)
    if ext in (".fvecs", ".ivecs", ".bvecs"):
        return _texmex(path, n_max)
    if ext == ".npy":
        a = np.load(path, mmap_mode="r")
        return a[:n_max] if n_max > 0 else a
    if ext in (".hdf5", ".h5"):
        import h5py
        a = h5py.File(path, "r")[key or "train"]
        return a[:n_max] if n_max > 0 else a[:]
    raise ValueError("不认识的格式:%s" % path)


def chunks(x, chunk=1 << 22):
    """按块产出 (起始行, float32 连续数组)。整数向量在这里转 float32。"""
    n = x.shape[0]
    for s in range(0, n, chunk):
        yield s, np.ascontiguousarray(x[s:min(n, s + chunk)], dtype=np.float32)


def write_bigann(path, a):
    """写 big-ann 格式(.fbin / .ibin),地面真值也用这个格式,和官方文件一致、可直接比对。"""
    a = np.ascontiguousarray(a)
    with open(path, "wb") as f:
        np.array(a.shape, np.int32).tofile(f)
        a.tofile(f)
