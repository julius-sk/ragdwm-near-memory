// mu_rabitq.cpp —— RaBitQ 两阶段扫描 kernel(XCENA MX1P)。
//
// 逐行对齐 rabitq_ref.py(而参考模型已对 FAISS 1.15.1 校验:第一阶段相对误差
// 2e-7、第二阶段 4e-7、top-10 一致)。改这里之前先改参考模型并重跑校验。
//
// 四个 kernel,对应 host 的四种 --mode:
//   I  rabitq_ints   只输出第一阶段的两个整数(dot, popcount)—— 单独验证位运算路径
//   E  rabitq_est1   第一阶段 1-bit 距离估计,全部 N 条
//   F  rabitq_full   第二阶段多 bit 完整距离,全部 N 条(只用于验证浮点路径)
//   T  rabitq_topk   真正的两阶段:每 task 在自己的片段上跑 FAISS 的平铺循环,
//                    输出本 task 的局部 top-k 和精算次数
//      rabitq_merge  单 task 把局部结果归并成全局 k 个 —— 只有这 k 个回传 host
//
// 为什么要有 I 模式。DRAN 项目记录过:手写标量浮点循环在本工具链 -O3 下算错
// (主机 0.00136,设备 -0.245)。RaBitQ 第一阶段是整数位运算 + 几次 fp32,
// 第二阶段是 d 维 fp32 标量循环。若 E/T 不对而 I 对,问题就在浮点,不在位运算。
//
// 数据布局(结构体数组拆开 SoA,与 prep_rabitq.py 一致):
//   signs  N x words uint64     符号码(sign bits),小端,尾部补零
//   fac1   N x 3 float          [or_minus_c_l2sqr, dp_multiplier, f_error]
//   excode N x exBytes uint8    附加码(ex code),小端位流
//   facx   N x 2 float          [f_add_ex, f_rescale_ex]
//   qblock QHeader(64 B) | planes(qb x words uint64) | rq(d float)
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

namespace
{
constexpr uint32_t K_MAX = 64;
constexpr uint32_t QB_MAX = 8;
constexpr uint32_t WORDS_MAX = 32;            // d <= 2048
constexpr uint32_t NONE = 0xFFFFFFFFu;

// 布局必须与 prep_rabitq.py 的 QHEADER 完全一致。
struct QHeader
{
    uint32_t d, words, qb, exBits, exBytes, k, rsv0, rsv1;
    float c1, c2, c34, qrToC, gError, cb, rsv2, rsv3;
};
static_assert(sizeof(QHeader) == 64, "QHeader 必须是 64 B");

#ifdef POPCNT_SWAR
inline uint32_t popcount64(uint64_t x)
{
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return static_cast<uint32_t>((x * 0x0101010101010101ull) >> 56);
}
#else
inline uint32_t popcount64(uint64_t x)
{
    return static_cast<uint32_t>(__builtin_popcountll(x));
}
#endif

// ceil 分块 —— 与 rabitq_ref.task_range、hamming_mu 的 myRange 同一规则。
inline void myRange(uint64_t n, uint64_t& begin, uint64_t& end)
{
    const uint64_t taskIdx = mu::getTaskIdx();
    const uint64_t taskCount = mu::getTaskCount();
    const uint64_t perTask = (n + taskCount - 1) / taskCount;
    begin = taskIdx * perTask;
    end = begin + perTask;
    if (end > n) end = n;
    if (begin > n) begin = n;
}

// 把 header 和位平面拷到栈上(<= 64 + 8*32*8 = 2112 B),内层循环不再碰设备内存里的查询。
struct Query
{
    QHeader h;
    uint64_t planes[QB_MAX * WORDS_MAX];
    const float* rq;          // 第二阶段的浮点查询,留在设备内存里按需读
};

inline bool loadQuery(const uint64_t* qblock, Query& q)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(qblock);
    __builtin_memcpy(&q.h, base, sizeof(QHeader));
    if (q.h.qb == 0 || q.h.qb > QB_MAX || q.h.words == 0 || q.h.words > WORDS_MAX ||
        q.h.k == 0 || q.h.k > K_MAX)
        return false;
    const uint64_t* p = reinterpret_cast<const uint64_t*>(base + sizeof(QHeader));
    for (uint32_t i = 0; i < q.h.qb * q.h.words; i++) q.planes[i] = p[i];
    q.rq = reinterpret_cast<const float*>(base + sizeof(QHeader) + 8ull * q.h.qb * q.h.words);
    return true;
}

// 第一阶段整数部分 —— 对应 rabitq_simd.h:160 bitwise_and_dot_product_with_popcount。
inline void stage1Ints(const uint64_t* s, const Query& q, uint64_t& dot, uint64_t& pc)
{
    dot = 0;
    pc = 0;
    for (uint32_t w = 0; w < q.h.words; w++)
    {
        const uint64_t x = s[w];
        pc += popcount64(x);
        for (uint32_t j = 0; j < q.h.qb; j++)
            dot += static_cast<uint64_t>(popcount64(q.planes[j * q.h.words + w] & x)) << j;
    }
}

// 第一阶段浮点部分 —— 对应 RaBitQuantizer.cpp:515 distance_to_code_1bit_impl(centered=false, L2)。
// 运算顺序照抄 FAISS:final_dot 先加 c1*dot,再加 c2*pc,再减 c34。
inline float stage1Est(uint64_t dot, uint64_t pc, const float* f1, const QHeader& h)
{
    float fd = 0.0f;
    fd += h.c1 * static_cast<float>(dot);
    fd += h.c2 * static_cast<float>(pc);
    fd -= h.c34;
    const float pre = (f1[0] + h.qrToC) - 2.0f * f1[1] * fd;
    return pre > 0.0f ? pre : 0.0f;
}

// 第二阶段 —— 对应 RaBitQUtils.h:383 compute_full_multibit_distance + rabitq_simd.h:351 ip_scalar。
// 每维码 = (符号位 << exBits) + 附加码;exBits <= 7,一个码最多跨两个字节,
// 且跨字节时下一个字节一定在 exBytes 之内,不会越界。
inline float stage2Full(const uint64_t* s, const uint8_t* ex, const float* fx, const Query& q)
{
    const uint32_t eb = q.h.exBits;
    const uint32_t mask = (1u << eb) - 1u;
    float acc = 0.0f;
    for (uint32_t i = 0; i < q.h.d; i++)
    {
        const int sb = static_cast<int>((s[i >> 6] >> (i & 63u)) & 1ull);
        const uint32_t bp = i * eb;
        const uint32_t byteIdx = bp >> 3;
        const uint32_t off = bp & 7u;
        uint32_t raw = ex[byteIdx];
        if (off + eb > 8u) raw |= static_cast<uint32_t>(ex[byteIdx + 1]) << 8;
        const int exv = static_cast<int>((raw >> off) & mask);
        acc += q.rq[i] * (static_cast<float>((sb << eb) + exv) + q.h.cb);
    }
    const float dist = q.h.qrToC + fx[0] + fx[1] * acc;
    return dist > 0.0f ? dist : 0.0f;
}

inline uint32_t floatBits(float f)
{
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return u;
}
}  // namespace

// ---------------------------------------------------------------------------
// I 模式:只输出整数。outInts = N x 2 uint32 [dot, popcount]
// ---------------------------------------------------------------------------
void rabitq_ints(const uint64_t* signs, const uint64_t* qblock, uint64_t numSigs, uint32_t* outInts)
{
    Query q;
    if (!loadQuery(qblock, q))
    {
        if (mu::getTaskIdx() == 0) mu::hostPrintf("[kernel] bad qblock header\n");
        return;
    }
    uint64_t begin, end;
    myRange(numSigs, begin, end);
    for (uint64_t i = begin; i < end; i++)
    {
        uint64_t dot, pc;
        stage1Ints(signs + i * q.h.words, q, dot, pc);
        outInts[2 * i] = static_cast<uint32_t>(dot);
        outInts[2 * i + 1] = static_cast<uint32_t>(pc);
        if (mu::getTaskIdx() == 0 && i == begin)
            mu::hostPrintf("[kernel] ints[%u] dot=%u pc=%u\n", (unsigned)i,
                           (unsigned)dot, (unsigned)pc);
    }
}

// ---------------------------------------------------------------------------
// E 模式:第一阶段距离估计。outDist = N float
// ---------------------------------------------------------------------------
void rabitq_est1(const uint64_t* signs, const float* fac1, const uint64_t* qblock,
                 uint64_t numSigs, float* outDist)
{
    Query q;
    if (!loadQuery(qblock, q))
    {
        if (mu::getTaskIdx() == 0) mu::hostPrintf("[kernel] bad qblock header\n");
        return;
    }
    uint64_t begin, end;
    myRange(numSigs, begin, end);
    for (uint64_t i = begin; i < end; i++)
    {
        uint64_t dot, pc;
        stage1Ints(signs + i * q.h.words, q, dot, pc);
        outDist[i] = stage1Est(dot, pc, fac1 + 3 * i, q.h);
        if (mu::getTaskIdx() == 0 && i == begin)
            mu::hostPrintf("[kernel] est1[%u] = 0x%08x\n", (unsigned)i, (unsigned)floatBits(outDist[i]));
    }
}

// ---------------------------------------------------------------------------
// F 模式:第二阶段完整距离,全部 N 条(验证用;真实搜索只对被选中的算)。outDist = N float
// ---------------------------------------------------------------------------
void rabitq_full(const uint64_t* signs, const uint8_t* excode, const float* facx,
                 const uint64_t* qblock, uint64_t numSigs, float* outDist)
{
    Query q;
    if (!loadQuery(qblock, q) || q.h.exBits == 0)
    {
        if (mu::getTaskIdx() == 0) mu::hostPrintf("[kernel] bad qblock or exBits==0\n");
        return;
    }
    uint64_t begin, end;
    myRange(numSigs, begin, end);
    for (uint64_t i = begin; i < end; i++)
    {
        outDist[i] = stage2Full(signs + i * q.h.words, excode + i * q.h.exBytes, facx + 2 * i, q);
        if (mu::getTaskIdx() == 0 && i == begin)
            mu::hostPrintf("[kernel] full[%u] = 0x%08x\n", (unsigned)i, (unsigned)floatBits(outDist[i]));
    }
}

// ---------------------------------------------------------------------------
// T 模式:两阶段 top-k。每 task 独立跑 FAISS 平铺循环(IndexRaBitQ.cpp:118-160)。
//   outRes   : taskCount x k x 2 uint32 —— 交错 (距离的位模式, id);空槽为 (inf, NONE)
//   outStats : taskCount x 2 uint32 —— [精算次数, 本 task 处理的向量数]
// 阈值 = 堆中最大值(未满时为 inf);替换第一个最大值 —— 与 rabitq_ref._scan_range 一字不差。
// ---------------------------------------------------------------------------
void rabitq_topk(const uint64_t* signs, const float* fac1, const uint8_t* excode,
                 const float* facx, const uint64_t* qblock, uint64_t numSigs,
                 uint32_t* outRes, uint32_t* outStats)
{
    Query q;
    const uint32_t taskIdx = mu::getTaskIdx();
    if (!loadQuery(qblock, q))
    {
        if (taskIdx == 0) mu::hostPrintf("[kernel] bad qblock header\n");
        return;
    }
    const uint32_t k = q.h.k;
    float hd[K_MAX];
    uint32_t hi[K_MAX];
    const float INF = __builtin_inff();
    for (uint32_t j = 0; j < k; j++) { hd[j] = INF; hi[j] = NONE; }

    uint64_t begin, end;
    myRange(numSigs, begin, end);
    uint32_t refined = 0;
    for (uint64_t i = begin; i < end; i++)
    {
        float thr = hd[0];
        uint32_t jmax = 0;
        for (uint32_t j = 1; j < k; j++)
            if (hd[j] > thr) { thr = hd[j]; jmax = j; }

        uint64_t dot, pc;
        const uint64_t* s = signs + i * q.h.words;
        const float* f1 = fac1 + 3 * i;
        stage1Ints(s, q, dot, pc);
        const float est = stage1Est(dot, pc, f1, q.h);
        float lb = est - f1[2] * q.h.gError;
        if (lb < 0.0f) lb = 0.0f;
        if (lb < thr)
        {
            refined++;
            const float full = (q.h.exBits == 0)
                ? est
                : stage2Full(s, excode + i * q.h.exBytes, facx + 2 * i, q);
            if (full < thr)
            {
                hd[jmax] = full;
                hi[jmax] = static_cast<uint32_t>(i);
            }
        }
    }

    uint32_t* my = outRes + static_cast<uint64_t>(taskIdx) * k * 2;
    for (uint32_t j = 0; j < k; j++)
    {
        my[2 * j] = floatBits(hd[j]);
        my[2 * j + 1] = hi[j];
    }
    outStats[2 * taskIdx] = refined;
    outStats[2 * taskIdx + 1] = static_cast<uint32_t>(end - begin);
    if (taskIdx == 0)
        mu::hostPrintf("[kernel] task0 range=[%u,%u) refined=%u\n",
                       (unsigned)begin, (unsigned)end, (unsigned)refined);
}

// ---------------------------------------------------------------------------
// T 模式第二步:单 task 归并。把 numTasks x k 个局部结果归并成全局 k 个。
//   outFinal : k x 2 uint32 (距离位模式, id),顺序不定,host 排序
// 不做这一步,每查询要回传 numTasks x k x 8 B(2816 x 10 时约 225 KB),
// 违背"只回传 k 个 id"。比较用 (距离, id) 字典序 —— 与 rabitq_ref.merge 的
// lexsort 一致,打平时结果也确定。
// ---------------------------------------------------------------------------
void rabitq_merge(const uint32_t* res, uint64_t numTasks, uint64_t k, uint32_t* outFinal)
{
    if (mu::getTaskIdx() != 0) return;
    if (k == 0 || k > K_MAX) { mu::hostPrintf("[kernel] merge: bad k\n"); return; }
    float bd[K_MAX];
    uint32_t bi[K_MAX];
    const float INF = __builtin_inff();
    for (uint64_t j = 0; j < k; j++) { bd[j] = INF; bi[j] = NONE; }

    const uint64_t total = numTasks * k;
    for (uint64_t e = 0; e < total; e++)
    {
        const uint32_t id = res[2 * e + 1];
        if (id == NONE) continue;
        float dv;
        __builtin_memcpy(&dv, &res[2 * e], 4);
        // 找当前最差的一槽:(距离, id) 字典序最大
        uint64_t jw = 0;
        for (uint64_t j = 1; j < k; j++)
            if (bd[j] > bd[jw] || (bd[j] == bd[jw] && bi[j] > bi[jw])) jw = j;
        if (dv < bd[jw] || (dv == bd[jw] && id < bi[jw]))
        {
            bd[jw] = dv;
            bi[jw] = id;
        }
    }
    for (uint64_t j = 0; j < k; j++)
    {
        outFinal[2 * j] = floatBits(bd[j]);
        outFinal[2 * j + 1] = bi[j];
    }
}

// ===========================================================================
// G 模式:全局阈值 τ 的两遍扫描 —— 设备上真正该用的两阶段做法
//
// T 模式(rabitq_topk)让每个 task 各自维护堆,是 FAISS 循环的直接并行化。但 2816 个
// task 时每个 task 只分到很少的向量,堆要先填满 k 个才有阈值,结果几乎全部被精算:
// FiQA(5.7 万条)上精算 90%,第二阶段字节是第一阶段的 2.3 倍。
// G 模式先求全局 τ = 所有向量上界 ub = est + f_error*g 中第 k 小的值:若误差界成立,
// 至少 k 条的真实距离 <= τ,下界 >= τ 的向量不可能进 top-k。参考模型在 FiQA 上
// 精算 0.71%(FAISS 单线程顺序扫描是 0.79%),召回不变。
//   rabitq_bounds  每 task:写 lb[i];本段最小的 k 个 ub -> outUB
//   rabitq_tau     单 task:numTasks x k 个 ub 中第 k 小 -> outTau[0]
//   rabitq_refine  每 task:只读 lb(4 B/条),lb < min(τ, 本地堆最大值) 才精算
// 之后用 rabitq_merge 归并。与 rabitq_ref.two_stage_global 一字不差。
// ===========================================================================
namespace
{
// buf 中保留最小的 k 个(未满为 inf);替换第一个最大值 —— 同 rabitq_ref._smallest_k_max
inline void keepSmallest(float* buf, uint32_t k, float v)
{
    uint32_t jmax = 0;
    for (uint32_t j = 1; j < k; j++)
        if (buf[j] > buf[jmax]) jmax = j;
    if (v < buf[jmax]) buf[jmax] = v;
}
}  // namespace

void rabitq_bounds(const uint64_t* signs, const float* fac1, const uint64_t* qblock,
                   uint64_t numSigs, float* outLB, float* outUB)
{
    Query q;
    const uint32_t taskIdx = mu::getTaskIdx();
    if (!loadQuery(qblock, q))
    {
        if (taskIdx == 0) mu::hostPrintf("[kernel] bad qblock header\n");
        return;
    }
    const uint32_t k = q.h.k;
    float buf[K_MAX];
    const float INF = __builtin_inff();
    for (uint32_t j = 0; j < k; j++) buf[j] = INF;

    uint64_t begin, end;
    myRange(numSigs, begin, end);
    for (uint64_t i = begin; i < end; i++)
    {
        uint64_t dot, pc;
        const float* f1 = fac1 + 3 * i;
        stage1Ints(signs + i * q.h.words, q, dot, pc);
        const float est = stage1Est(dot, pc, f1, q.h);
        const float adj = f1[2] * q.h.gError;
        float lb = est - adj;
        if (lb < 0.0f) lb = 0.0f;
        outLB[i] = lb;
        keepSmallest(buf, k, est + adj);
    }
    float* my = outUB + static_cast<uint64_t>(taskIdx) * k;
    for (uint32_t j = 0; j < k; j++) my[j] = buf[j];
}

void rabitq_tau(const float* ub, uint64_t numTasks, uint64_t k, float* outTau)
{
    if (mu::getTaskIdx() != 0) return;
    if (k == 0 || k > K_MAX) { mu::hostPrintf("[kernel] tau: bad k\n"); return; }
    float buf[K_MAX];
    const float INF = __builtin_inff();
    for (uint64_t j = 0; j < k; j++) buf[j] = INF;
    const uint64_t total = numTasks * k;
    for (uint64_t e = 0; e < total; e++) keepSmallest(buf, (uint32_t)k, ub[e]);
    float tau = buf[0];
    for (uint64_t j = 1; j < k; j++)
        if (buf[j] > tau) tau = buf[j];
    outTau[0] = tau;
    mu::hostPrintf("[kernel] tau = 0x%08x\n", (unsigned)floatBits(tau));
}

void rabitq_refine(const uint64_t* signs, const uint8_t* excode, const float* facx,
                   const uint64_t* qblock, const float* lb, const float* tauPtr,
                   uint64_t numSigs, uint32_t* outRes, uint32_t* outStats)
{
    Query q;
    const uint32_t taskIdx = mu::getTaskIdx();
    if (!loadQuery(qblock, q) || q.h.exBits == 0)
    {
        if (taskIdx == 0) mu::hostPrintf("[kernel] bad qblock or exBits==0\n");
        return;
    }
    const uint32_t k = q.h.k;
    const float tau = tauPtr[0];
    float hd[K_MAX];
    uint32_t hi[K_MAX];
    const float INF = __builtin_inff();
    for (uint32_t j = 0; j < k; j++) { hd[j] = INF; hi[j] = NONE; }

    uint64_t begin, end;
    myRange(numSigs, begin, end);
    uint32_t refined = 0;
    for (uint64_t i = begin; i < end; i++)
    {
        float hmax = hd[0];
        uint32_t jmax = 0;
        for (uint32_t j = 1; j < k; j++)
            if (hd[j] > hmax) { hmax = hd[j]; jmax = j; }
        const float thr = (tau < hmax) ? tau : hmax;
        if (lb[i] < thr)
        {
            refined++;
            const float full = stage2Full(signs + i * q.h.words, excode + i * q.h.exBytes,
                                          facx + 2 * i, q);
            if (full < hmax)
            {
                hd[jmax] = full;
                hi[jmax] = static_cast<uint32_t>(i);
            }
        }
    }
    uint32_t* my = outRes + static_cast<uint64_t>(taskIdx) * k * 2;
    for (uint32_t j = 0; j < k; j++)
    {
        my[2 * j] = floatBits(hd[j]);
        my[2 * j + 1] = hi[j];
    }
    outStats[2 * taskIdx] = refined;
    outStats[2 * taskIdx + 1] = static_cast<uint32_t>(end - begin);
}

MU_KERNEL_ADD(rabitq_ints);
MU_KERNEL_ADD(rabitq_est1);
MU_KERNEL_ADD(rabitq_full);
MU_KERNEL_ADD(rabitq_topk);
MU_KERNEL_ADD(rabitq_merge);
MU_KERNEL_ADD(rabitq_bounds);
MU_KERNEL_ADD(rabitq_tau);
MU_KERNEL_ADD(rabitq_refine);
