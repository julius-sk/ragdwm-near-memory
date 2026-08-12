// mu_hamming.cpp — 256 位签名的 Hamming 扫描 kernel(XCENA MX1P)。
//
// 负载:query 签名对 N 个存储签名做 XOR + POPCOUNT,得 0..256 的距离。
// 每签名 4 x uint64 = 32B 连续存放,每 task 扫一段连续签名 —— DRAM 最擅长的顺序读。
//
// popcount 有两种实现,用 -DPOPCNT_SWAR 切换:
//   默认        __builtin_popcountll(RISC-V 有 Zbb 扩展时映射到 cpop 指令)
//   POPCNT_SWAR 经典 5 步掩码分治法,不依赖硬件指令
// 两者必须都通过 numpy 逐位验证 —— DRAN 项目中手写标量循环曾在本工具链 -O3 下
// 产生错误结果,不能假定整数路径必然正确。
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

namespace
{
constexpr int WORDS = 4;              // 每签名 4 x uint64 = 256 位
constexpr int BITS = WORDS * 64;      // 256

#ifdef POPCNT_SWAR
// SWAR 位并行 popcount:5 步掩码分治,不依赖硬件指令。
inline int popcount64(uint64_t x)
{
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return static_cast<int>((x * 0x0101010101010101ull) >> 56);
}
#else
inline int popcount64(uint64_t x)
{
    return __builtin_popcountll(x);
}
#endif

// 一个签名(4 x uint64)与 query 的 Hamming 距离。
inline int hammingDist(const uint64_t* s, const uint64_t* q)
{
    int d = 0;
    for (int w = 0; w < WORDS; w++)
    {
        d += popcount64(s[w] ^ q[w]);
    }
    return d;
}

// 把 [0,numSigs) 分给 taskCount 个 task,规则是 **ceil 分块**:
// 每 task 取 ceil(n/taskCount) 个,末尾的 task 分到的更少、甚至为空
// (不是"最后一个吃掉余数" —— 两种约定给出不同划分,n=10/tc=4 时
//  ceil 分块是 [0,3)[3,6)[6,9)[9,10),余数给最后是 [0,2)[2,4)[4,6)[6,10))。
// 必须与 model_kernel.py 的 _task_range 保持同一规则,否则模型与设备静默分歧。
inline void myRange(uint64_t numSigs, uint64_t& begin, uint64_t& end)
{
    const uint64_t taskIdx = mu::getTaskIdx();
    const uint64_t taskCount = mu::getTaskCount();
    const uint64_t perTask = (numSigs + taskCount - 1) / taskCount;
    begin = taskIdx * perTask;
    end = begin + perTask;
    if (end > numSigs) end = numSigs;
    if (begin > numSigs) begin = numSigs;
}
}  // namespace

// ---------------------------------------------------------------------------
// B 组 —— 只算距离,全部写出。host 负责选 top-k。
//
//   sigs     : N x 4 uint64,连续(设备内存,已 preload)
//   query    : 4 uint64
//   numSigs  : N
//   outDists : N 个 int32,值域 [0,256]
//
// 每 task 只写自己那段,无跨 task 竞争。
// ---------------------------------------------------------------------------
void hamming_scan_dists(const uint64_t* sigs,
                        const uint64_t* query,
                        uint64_t numSigs,
                        int32_t* outDists)
{
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    uint64_t q[WORDS];
    for (int w = 0; w < WORDS; w++) q[w] = query[w];

    if (mu::getTaskIdx() == 0)
    {
        mu::hostPrintf("[kernel] taskCount=%u range=[%u,%u) q0=0x%08x%08x\n",
                       (unsigned)mu::getTaskCount(),
                       (unsigned)begin, (unsigned)end,
                       (unsigned)(q[0] >> 32), (unsigned)(q[0] & 0xFFFFFFFFu));
    }

    for (uint64_t i = begin; i < end; i++)
    {
        const int d = hammingDist(sigs + i * WORDS, q);
        outDists[i] = static_cast<int32_t>(d);
        if (mu::getTaskIdx() == 0 && i == begin)
        {
            mu::hostPrintf("[kernel] dist[%u] = %d (raw 0x%08x)\n",
                           (unsigned)i, d, (unsigned)d);
        }
    }
}

// ---------------------------------------------------------------------------
// C 组 stage 1 —— 扫描 + 每 task 257 桶直方图 + 候选收集。
//
//   coarseThresh : 粗阈值,dist <= 此值的候选收进分片
//   candCap      : 每 task 候选**条目数**上限(每条目 2 个 uint32)
//   outHist      : taskCount * 257 uint32
//   outCand      : taskCount * candCap * 2 uint32 —— 交错 (id, dist)
//   outMeta      : taskCount * 2 uint32 —— [0]=candCount [1]=overflow
//
// ★ 候选必须连距离一起存。只存 id 的话 merge 无法按 exactThresh 过滤,
//   粗阈值偏大时会静默返回宽带里的任意 id(实测:返回距离 145,真实第k名 113,
//   而成功门仍然通过)。详见 Task 2b 的模型与回归用例。
//
// 直方图放在堆上而非栈上:257 * 4 = 1028 B 虽然放得下 64 KB 栈,但显式用
// 传入的设备内存分片更省栈,也便于 merge kernel 直接读。
// ---------------------------------------------------------------------------
void hamming_scan_hist(const uint64_t* sigs,
                       const uint64_t* query,
                       uint64_t numSigs,
                       int32_t coarseThresh,
                       int32_t candCap,
                       uint32_t* outHist,
                       uint32_t* outCand,
                       uint32_t* outMeta)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    uint64_t q[WORDS];
    for (int w = 0; w < WORDS; w++) q[w] = query[w];

    uint32_t* myHist = outHist + (uint64_t)taskIdx * (BITS + 1);
    uint32_t* myCand = outCand + (uint64_t)taskIdx * (uint64_t)candCap * 2;  // 每条目 2 个 uint32
    uint32_t* myMeta = outMeta + (uint64_t)taskIdx * 2;

    for (int b = 0; b <= BITS; b++) myHist[b] = 0u;

    uint32_t nCand = 0;
    uint32_t overflow = 0;

    for (uint64_t i = begin; i < end; i++)
    {
        const int d = hammingDist(sigs + i * WORDS, q);
        myHist[d]++;                       // 距离有界 -> 计数排序的基础
        if (d <= coarseThresh)
        {
            if (nCand < (uint32_t)candCap)
            {
                myCand[2 * nCand] = (uint32_t)i;      // id
                myCand[2 * nCand + 1] = (uint32_t)d;  // ★ 距离必须一起存
                nCand++;
            }
            else overflow = 1u;            // 记录但不中断:直方图仍然完整
        }
    }

    myMeta[0] = nCand;
    myMeta[1] = overflow;

    if (taskIdx == 0)
    {
        mu::hostPrintf("[kernel] task0 range=[%u,%u) nCand=%u overflow=%u\n",
                       (unsigned)begin, (unsigned)end, (unsigned)nCand, (unsigned)overflow);
    }
}

// ---------------------------------------------------------------------------
// C 组 stage 2 —— 单 task 归并。
//
// 1. 归并所有 task 的直方图
// 2. 从距离 0 累加,定位第 topK 名所在的桶 -> 精确阈值 exactThresh
// 3. 扫候选分片收集 id;若任一 task 溢出且 dists 非空,则退回扫 dists 补齐
//
// 必须单独一个 kernel:归并要看到所有 task 的直方图,而 Map 内无跨 task 同步。
// ---------------------------------------------------------------------------
void hamming_merge_topk(const uint32_t* hist,
                        const uint32_t* cand,
                        const uint32_t* meta,
                        int32_t taskCount,
                        int32_t candCap,
                        int32_t topK,
                        const int32_t* dists,
                        uint32_t* outIds)
{
    if (mu::getTaskIdx() != 0) return;

    // --- 1. 归并直方图 ---
    uint32_t total[BITS + 1];
    for (int b = 0; b <= BITS; b++) total[b] = 0u;
    for (int t = 0; t < taskCount; t++)
    {
        const uint32_t* h = hist + (uint64_t)t * (BITS + 1);
        for (int b = 0; b <= BITS; b++) total[b] += h[b];
    }

    // --- 2. 定位第 topK 名的精确距离阈值 ---
    int exactThresh = BITS;
    uint64_t cum = 0;
    for (int b = 0; b <= BITS; b++)
    {
        cum += total[b];
        if (cum >= (uint64_t)topK) { exactThresh = b; break; }
    }

    // --- 3. 检查是否有 task 溢出 ---
    int anyOverflow = 0;
    for (int t = 0; t < taskCount; t++)
    {
        if (meta[(uint64_t)t * 2 + 1]) { anyOverflow = 1; break; }
    }

    mu::hostPrintf("[kernel] exactThresh=%d anyOverflow=%d\n", exactThresh, anyOverflow);

    // ★ 只收 dist <= exactThresh 的候选。没有这个过滤,粗阈值偏大时会返回
    //   宽带里的任意 id —— 已复现的静默错误(返回距离 145,真实第k名 113)。
    //   有了它,每个返回 id 必然满足 d <= 第k名距离,溢出便不再能产生错误结果。
    int n = 0;
    for (int t = 0; t < taskCount && n < topK; t++)
    {
        const uint32_t cnt = meta[(uint64_t)t * 2];
        const uint32_t* c = cand + (uint64_t)t * (uint64_t)candCap * 2;
        for (uint32_t j = 0; j < cnt && n < topK; j++)
        {
            if ((int)c[2 * j + 1] <= exactThresh) outIds[n++] = c[2 * j];
        }
    }

    // 不足 topK 时用哨兵填充,host 可据此判断收集是否完整
    const int collected = n;
    for (; n < topK; n++) outIds[n] = 0xFFFFFFFFu;

    // outIds[topK] = exactThresh, outIds[topK+1] = anyOverflow, outIds[topK+2] = collected
    // (host 端为 outIds 多分配了 3 个槽位)
    outIds[topK]     = (uint32_t)exactThresh;
    outIds[topK + 1] = (uint32_t)anyOverflow;
    outIds[topK + 2] = (uint32_t)collected;

    (void)dists;  // 保留参数位以便将来做设备内重扫;当前重跑由 host 驱动
}

MU_KERNEL_ADD(hamming_scan_dists);
MU_KERNEL_ADD(hamming_scan_hist);
MU_KERNEL_ADD(hamming_merge_topk);
