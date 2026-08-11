// mu_hamming.cpp — 256 位签名的 Hamming 扫描 kernel(XCENA MX1P)。
//
// 负载:query 签名对 N 个存储签名做 XOR + POPCOUNT,得 0..256 的距离,选 top-k。
// 每签名 4 x uint64 = 32B 连续存放,每 task 扫一段连续签名 —— DRAM 最擅长的顺序读。
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

// 空骨架:仅验证工具链能编译并产出 .mubin。
void hamming_scan(const uint64_t* sigs, uint64_t numSigs, uint32_t* out)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    if (taskIdx == 0)
    {
        mu::hostPrintf("[kernel] skeleton alive: numSigs=%u\n", (unsigned)numSigs);
        out[0] = 1u;
    }
}

MU_KERNEL_ADD(hamming_scan);
