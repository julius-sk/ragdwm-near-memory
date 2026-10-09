"""rabitq_ref.py —— RaBitQ 两阶段扫描的参考模型(reference model),逐行对齐 FAISS 1.15.1。

为什么要有它。MX1P 的 kernel 只能盲写(开发机没有 C++ 编译器,也没有卡),
上了 solab-h1 第一次跑,唯一能判断对错的就是一份**可信的**参考答案。这份参考
答案本身先在开发机上和 FAISS 逐项比对(test_ref_vs_faiss.py),kernel 再逐行照
着本文件写,设备输出拿回来和本文件比(verify_rabitq.py)。三层链条缺一层,
错误就会无声地进入论文。

对齐的 FAISS 源码(v1.15.1,行号见注释):
  查询因子(query factors)    RaBitQUtils.cpp  compute_query_factors
  查询量化                    rabitq_simd.h    quantize_query_values / round_clamped_byte_scalar
  位平面重排(bit-plane)     rabitq_simd.h:229 rearrange_bit_planes
  第一阶段 1-bit 估计         RaBitQuantizer.cpp:515 distance_to_code_1bit_impl(centered=false)
                              rabitq_simd.h:160 bitwise_and_dot_product_with_popcount
  精算判定                    RaBitQUtils.h:300 should_refine_candidate
  第二阶段多 bit 距离         RaBitQUtils.h:383 compute_full_multibit_distance
                              rabitq_simd.h:351 ip_scalar

只覆盖本项目用到的配置:METRIC_L2、centered=false(FAISS 默认)、qb>0。

设备上的数据布局与 FAISS 不同,是**结构体数组拆开(SoA, structure of arrays)**:
  signs  N x W  uint64   符号码,W = ceil(ceil(d/8)/8),尾部补零
  fac1   N x 3  float32  [or_minus_c_l2sqr, dp_multiplier, f_error]   第一阶段要读的
  excode N x E  uint8    附加码(ex code),E = ceil(d*ex_bits/8)      第二阶段随机读
  facx   N x 2  float32  [f_add_ex, f_rescale_ex]
FAISS 把这些按每条向量连续存放(AoS);第一阶段只需前 60 B,却要把整条 212 B
读进来。拆开后第一阶段是纯顺序读,第二阶段只碰被选中的向量。
"""
import numpy as np

try:
    from numba import njit
except ImportError:          # numba 只是加速 replay;没有也能跑,只是慢
    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]

U32_NONE = np.uint32(0xFFFFFFFF)


# ---------------------------------------------------------------------------
# 数据库侧:从 FAISS IndexRaBitQ 的码字节里拆出 SoA
# ---------------------------------------------------------------------------
def split_codes(index):
    import faiss
    d, nb, N = index.d, int(index.rabitq.nb_bits), index.ntotal
    cs = index.code_size
    codes = faiss.vector_to_array(index.codes).reshape(N, cs)
    sb = (d + 7) // 8                      # 符号码字节数
    W = (sb + 7) // 8                      # 补到 uint64 整数个
    sbytes = np.zeros((N, W * 8), np.uint8)
    sbytes[:, :sb] = codes[:, :sb]
    signs = sbytes.view("<u8").reshape(N, W)           # 小端:与 FAISS 的 *(uint64_t*) 读法一致
    fac1 = np.zeros((N, 3), np.float32)
    ex_bits = nb - 1
    if ex_bits == 0:
        fac1[:, :2] = codes[:, sb:sb + 8].copy().view("<f4").reshape(N, 2)   # SignBitFactors
        excode = np.zeros((N, 0), np.uint8)
        facx = np.zeros((N, 2), np.float32)
    else:
        fac1[:] = codes[:, sb:sb + 12].copy().view("<f4").reshape(N, 3)      # SignBitFactorsWithError
        E = (d * ex_bits + 7) // 8
        excode = codes[:, sb + 12:sb + 12 + E].copy()
        facx = codes[:, sb + 12 + E:sb + 12 + E + 8].copy().view("<f4").reshape(N, 2)
    center = faiss.vector_to_array(index.center).astype(np.float32)
    return {"d": d, "nb": nb, "ex_bits": ex_bits, "N": N, "W": W,
            "signs": np.ascontiguousarray(signs), "fac1": fac1,
            "excode": np.ascontiguousarray(excode), "facx": facx, "center": center}


# ---------------------------------------------------------------------------
# 查询侧:compute_query_factors(centered=false, METRIC_L2)
# ---------------------------------------------------------------------------
def query_block(q, center, d, W, qb=4, ex_bits=0):
    f32 = np.float32
    q = np.asarray(q, f32)
    rq = (q - center).astype(f32)                       # rotated_q[i] = query[i] - centroid[i]
    # fvec_L2sqr 在 FAISS 里是 SIMD 求和,顺序不同 -> 只能做到 ulp 级一致,验证用容差
    qr_to_c = f32(np.sum((q.astype(np.float64) - center) ** 2))
    v_min, v_max = f32(rq.min()), f32(rq.max())
    max_code = (1 << qb) - 1
    delta = f32((v_max - v_min) / f32(max_code))
    if delta <= 0:
        qq = np.zeros(d, np.int64)
    else:
        x = (rq - v_min) * (f32(1.0) / delta)          # (rq[i] - v_min) * inv_delta,float32
        qq = np.where(x <= 0, 0, np.where(x >= max_code, max_code,
                                          (x + f32(0.5)).astype(np.int64)))
    sum_qq = int(qq.sum())
    inv_d_sqrt = f32(1.0) / f32(np.sqrt(f32(d)))
    c1 = f32(f32(2) * delta) * inv_d_sqrt
    c2 = f32(f32(2) * v_min) * inv_d_sqrt
    c34 = inv_d_sqrt * f32(f32(delta * f32(sum_qq)) + f32(f32(d) * v_min))
    # 位平面:第 j 个平面的第 i 位 = qq[i] 的第 j 位;字节 i/8 的第 i%8 位
    planes = np.zeros((qb, W * 8), np.uint8)
    for j in range(qb):
        bits = ((qq >> j) & 1).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little")
        planes[j, :packed.shape[0]] = packed
    planes = planes.view("<u8").reshape(qb, W)
    return {"qb": qb, "ex_bits": ex_bits, "c1": c1, "c2": c2, "c34": c34,
            "qr_to_c": qr_to_c, "g_error": f32(np.sqrt(qr_to_c)),
            "cb": f32(-(float(1 << ex_bits) - 0.5)), "planes": np.ascontiguousarray(planes),
            "rq": rq, "qq": qq}


# ---------------------------------------------------------------------------
# 第一阶段:1-bit 估计(全部 N 条)
# ---------------------------------------------------------------------------
def stage1_ints(signs, planes):
    """整数部分:dot = sum_j popcount(plane_j & s) << j,pc = popcount(s)。
    kernel 的诊断模式 I 输出的正是这两个数 —— 先单独验证整数路径。"""
    dot = np.zeros(signs.shape[0], np.int64)
    for j in range(planes.shape[0]):
        dot += np.bitwise_count(signs & planes[j]).sum(1).astype(np.int64) << j
    pc = np.bitwise_count(signs).sum(1).astype(np.int64)
    return dot, pc


def stage1_est(signs, fac1, qblk):
    f32 = np.float32
    dot, pc = stage1_ints(signs, qblk["planes"])
    final = f32(0) + qblk["c1"] * dot.astype(f32)       # final_dot += c1 * dot_product
    final = final + qblk["c2"] * pc.astype(f32)          # final_dot += c2 * popcount
    final = final - qblk["c34"]                          # final_dot -= c34
    pre = (fac1[:, 0] + qblk["qr_to_c"]) - (f32(2) * fac1[:, 1]) * final
    return np.maximum(f32(0), pre).astype(f32)


# ---------------------------------------------------------------------------
# 第二阶段:多 bit 完整距离
# ---------------------------------------------------------------------------
def code_values(signs, excode, d, ex_bits):
    """每维的 nb 位整数码 = (符号位 << ex_bits) + 附加码。"""
    N = signs.shape[0]
    sb = np.unpackbits(signs.view(np.uint8), axis=1, bitorder="little")[:, :d].astype(np.int64)
    if ex_bits == 0:
        return sb
    eb = np.unpackbits(excode, axis=1, bitorder="little")[:, :d * ex_bits]
    eb = eb.reshape(N, d, ex_bits).astype(np.int64)
    ex = (eb << np.arange(ex_bits)).sum(2)
    return (sb << ex_bits) + ex


def stage2_full(signs, excode, facx, qblk, d, chunk=1 << 18):
    """分块解码:一次解码全部码要 N x d 个 int64(884 万 x 384 时约 27 GB)。"""
    ex_bits = qblk["ex_bits"]
    rq = qblk["rq"].astype(np.float64)
    out = np.empty(signs.shape[0], np.float32)
    for s in range(0, signs.shape[0], chunk):
        e = min(s + chunk, signs.shape[0])
        vals = code_values(signs[s:e], excode[s:e], d, ex_bits).astype(np.float64)
        ex_ip = (vals + float(qblk["cb"])) @ rq
        dist = float(qblk["qr_to_c"]) + facx[s:e, 0].astype(np.float64) + facx[s:e, 1] * ex_ip
        out[s:e] = np.maximum(0.0, dist)
    return out


# ---------------------------------------------------------------------------
# 两阶段扫描:每个 task 在自己的片段上跑 FAISS 的平铺循环(IndexRaBitQ.cpp:118-160)
# ---------------------------------------------------------------------------
def task_range(n, task_count, t):
    """ceil 分块 —— 与 kernel 的 myRange、hamming_mu 的 _task_range 同一规则。"""
    per = (n + task_count - 1) // task_count
    b = min(t * per, n)
    return b, min(b + per, n)


@njit(cache=True)
def _scan_range(est1, full, ferr, g, b, e, k, out_d, out_i):
    """返回本 task 的精算次数;堆写进 out_d / out_i(未填满的位置保持 inf / NONE)。
    规则与 kernel 一字不差:阈值 = 堆中最大值(未满时为 inf);替换第一个最大值。"""
    refined = 0
    for i in range(b, e):
        thr = out_d[0]
        jmax = 0
        for j in range(1, k):
            if out_d[j] > thr:
                thr = out_d[j]
                jmax = j
        lb = est1[i] - ferr[i] * g
        if lb < 0.0:
            lb = 0.0
        if lb < thr:
            refined += 1
            if full[i] < thr:
                out_d[jmax] = full[i]
                out_i[jmax] = i
    return refined


def two_stage(est1, full, ferr, g, k, task_count):
    """返回 (每 task 结果 [T,k] dist/id, 每 task 精算数 [T], 合并后的全局 top-k)。"""
    n = est1.shape[0]
    T = task_count
    res_d = np.full((T, k), np.inf, np.float32)
    res_i = np.full((T, k), U32_NONE, np.uint32)
    refined = np.zeros(T, np.int64)
    e1 = est1.astype(np.float32)
    fu = full.astype(np.float32)
    fe = ferr.astype(np.float32)
    for t in range(T):
        b, e = task_range(n, T, t)
        refined[t] = _scan_range(e1, fu, fe, np.float32(g), b, e, k, res_d[t], res_i[t])
    return res_d, res_i, refined, merge(res_d, res_i, k)


# ---------------------------------------------------------------------------
# G 模式:全局阈值 τ 的两遍扫描(设备上真正该用的)
#
# 为什么。上面的 two_stage 让每个 task 各自维护堆 —— 这是 FAISS 循环的直接并行化,
# 但 2816 个 task 时每个 task 只分到很少的向量,堆要先填满 k 个才有阈值,结果几乎
# 全部被精算:FiQA(5.7 万条)上精算 90%,第二阶段字节是第一阶段的 2.3 倍。
# 全局 τ:先算出所有向量的上界 ub = est + f_error*g,取第 k 小的作为 τ。若误差界成立,
# 至少有 k 条的真实距离 <= τ,所以下界 >= τ 的向量不可能进 top-k。FiQA 上精算降到
# 0.71%(比 FAISS 单线程顺序扫描的 0.79% 还少),召回不变。
#
# 设备上分三个 kernel:
#   bounds  每 task:写出每条的下界 lb;记下本段最小的 k 个 ub
#   tau     单 task:所有 task 的 k 个 ub 里取第 k 小 -> τ
#   refine  每 task:只读 lb(4 B/条),lb < min(τ, 本地堆最大值) 才精算
# ---------------------------------------------------------------------------
@njit(cache=True)
def _smallest_k_max(vals, b, e, k, buf):
    """把 vals[b:e] 中最小的 k 个放进 buf(未满为 inf),返回 buf 的最大值。替换规则同堆。"""
    for j in range(k):
        buf[j] = np.inf
    for i in range(b, e):
        jmax = 0
        for j in range(1, k):
            if buf[j] > buf[jmax]:
                jmax = j
        if vals[i] < buf[jmax]:
            buf[jmax] = vals[i]
    m = buf[0]
    for j in range(1, k):
        if buf[j] > m:
            m = buf[j]
    return m


@njit(cache=True)
def _refine_range(lb, full, tau, b, e, k, out_d, out_i):
    refined = 0
    for i in range(b, e):
        hmax = out_d[0]
        jmax = 0
        for j in range(1, k):
            if out_d[j] > hmax:
                hmax = out_d[j]
                jmax = j
        thr = tau if tau < hmax else hmax
        if lb[i] < thr:
            refined += 1
            if full[i] < hmax:
                out_d[jmax] = full[i]
                out_i[jmax] = i
    return refined


def bounds(est1, ferr, g):
    """与 kernel rabitq_bounds 相同的 float32 运算。"""
    f32 = np.float32
    e = est1.astype(f32)
    adj = ferr.astype(f32) * f32(g)
    return np.maximum(f32(0), e - adj), e + adj


def two_stage_global(est1, full, ferr, g, k, task_count):
    """返回 (每 task 结果, 每 task 精算数, 合并后的全局 top-k, τ)。"""
    n = est1.shape[0]
    T = task_count
    lb, ub = bounds(est1, ferr, g)
    # tau:每 task 本段最小的 k 个 ub -> 合起来再取最小的 k 个 -> 其最大值
    per = np.full((T, k), np.inf, np.float32)
    buf = np.empty(k, np.float32)
    for t in range(T):
        b, e = task_range(n, T, t)
        _smallest_k_max(ub, b, e, k, buf)
        per[t] = buf
    flat = per.ravel()
    tau = np.float32(_smallest_k_max(flat, 0, flat.shape[0], k, np.empty(k, np.float32)))
    res_d = np.full((T, k), np.inf, np.float32)
    res_i = np.full((T, k), U32_NONE, np.uint32)
    refined = np.zeros(T, np.int64)
    fu = full.astype(np.float32)
    for t in range(T):
        b, e = task_range(n, T, t)
        refined[t] = _refine_range(lb, fu, tau, b, e, k, res_d[t], res_i[t])
    return res_d, res_i, refined, merge(res_d, res_i, k), tau


def merge(res_d, res_i, k):
    """主机侧归并:按 (距离, id) 排序取前 k,空槽丢弃。"""
    d = res_d.ravel()
    i = res_i.ravel()
    keep = i != U32_NONE
    d, i = d[keep], i[keep]
    order = np.lexsort((i, d))[:k]
    return d[order], i[order]
