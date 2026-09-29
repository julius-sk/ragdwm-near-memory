// mu_roofline.cpp —— MX1P 的内存带宽 roofline 探针。
//
// 为什么需要它。RESULTS.md 记录设备扫描 27.9 GB/s(B 组)/ 35.3 GB/s(C 组),
// 主机 51.8 GB/s。但这三个数都不能回答唯一重要的那个问题:
//
//     设备的 35.3 GB/s 是**这块卡的带宽极限**,还是**这个 kernel 的实现极限**?
//
// 差别决定项目走向。若设备天花板就在 36 GB/s 附近,那么再怎么优化 kernel 也追不上
// 主机,近内存这条路可以就此结束;若天花板在 100 GB/s,那 35.3 只用了三分之一,
// 差距是实现问题,值得继续投入。
//
// 本文件给出这个上限:与 hamming_scan_dists **读完全相同的字节、相同的顺序、
// 相同的 task 划分**,但把 popcount 换成 XOR 累加,并且不写逐元素输出。
// 任何真实扫描 kernel 都不可能快过它。
//
// 提供两个探针,差别只在输出量,用来把"读"和"写"拆开:
//
//   hamming_bw_read    只读,每 task 写一个 uint64。纯读带宽。
//   hamming_bw_readwrite  额外写 N 个 int32,与 B 组的输出量一致。
//
// 这个拆分本身就能解释一件事:B 组 27.9 GB/s 与 C 组 35.3 GB/s 的差距,很可能
// 就是 B 组在 N=100M 时多写的 400 MB 距离数组。两个探针一比即知,不必猜。
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

namespace
{
constexpr int WORDS = 4;   // 每签名 4 x uint64 = 256 位,与 mu_hamming.cpp 一致

// 与 mu_hamming.cpp 的 myRange 同一规则(ceil 分块)。
// 这里刻意复制而不是共享:两个文件各自编译进同一个 .mubin,匿名命名空间不互通。
// 划分规则必须与真实 kernel 一致 —— 否则两者读取的内存访问模式不同,
// 测出来的"上限"就不是真实 kernel 的上限。
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
// 纯读带宽上限。
//
//   sigs     : N x 4 uint64,连续(设备内存,已 preload)
//   numSigs  : N
//   outAcc   : taskCount 个 uint64,每 task 写一个
//
// 累加值必须写出去,否则 -O3 会把整个循环删掉,测出来是无限带宽 —— 这类基准
// 最经典的失效方式,而且它的症状("数字好得离谱")和成功长得一模一样。
// ---------------------------------------------------------------------------
void hamming_bw_read(const uint64_t* sigs, uint64_t numSigs, uint64_t* outAcc)
{
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    uint64_t acc = 0;
    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t* s = sigs + i * WORDS;
        acc ^= s[0] ^ s[1] ^ s[2] ^ s[3];
    }
    outAcc[mu::getTaskIdx()] = acc;
}

// ---------------------------------------------------------------------------
// 读 + 逐元素写,输出量与 B 组一致(N 个 int32)。
//
// 与 hamming_bw_read 的差值 = B 组那 400 MB(N=100M 时)输出写入的代价。
// 写的是 XOR 折叠后的低 9 位,值域与真实距离一致([0,511] 截断到 int32),
// 这样写入的数据模式不至于因为全零而被内存控制器特殊对待。
// ---------------------------------------------------------------------------
void hamming_bw_readwrite(const uint64_t* sigs, uint64_t numSigs, int32_t* outVals)
{
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    for (uint64_t i = begin; i < end; i++)
    {
        const uint64_t* s = sigs + i * WORDS;
        const uint64_t x = s[0] ^ s[1] ^ s[2] ^ s[3];
        outVals[i] = static_cast<int32_t>(x & 0x1FF);
    }
}
