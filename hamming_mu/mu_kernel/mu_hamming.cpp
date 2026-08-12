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

MU_KERNEL_ADD(hamming_scan_dists);
