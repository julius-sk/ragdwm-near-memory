// preload_probe.cpp —— 量出这张卡上 MU 核真正能用的容量:Preload Region 有多少个 1 GB 槽位,
// 以及一次能 preload 多大的连续一块。
//
// 为什么要量。在 InfiniteMemory 计算版固件上,MU 核只能读 Preload Region(实战手册 §11.2:
// 读没 preload 的内存返回 Success 但结果是错的)。卡标称的 1.8 TB 是 SSD 撑起来的容量,
// 不是 MU 能算的容量。我们所有「库能放多大」的论断都卡在这个数上,而文档只说「大约一半
// DRAM」,h1 上没人量过。
//
// 做法(不写任何数据,不会冲掉别人的缓存内容,但会占用 Preload 槽位,跑之前确认卡空闲):
//   第 1 步  逐个分配 1 GB 并 preload,直到返回 ResourceExhaust —— 得到槽位总数 S
//   第 2 步  全部 unload / release
//   第 3 步  一次分配 S GB 并 preload;失败就 S-1、S-2 …… —— 得到最大连续块
//            (文档:preload 需要一段连续的 1 GB 槽位,槽位碎片化时会 ResourceExhaust)
//
// Build: g++ -O2 -std=c++17 preload_probe.cpp -o preload_probe -I/usr/local/pxl/include -lpxl
//        (路径以 SDK 为准;run_bench.sh 的 build 步骤会处理)
// Run:   ./preload_probe [--device 0] [--max 512]

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "pxl/pxl.hpp"

namespace
{
constexpr size_t GB = 1ull << 30;

const char* statusName(pxl::MemoryStatus s)
{
    switch (s)
    {
    case pxl::MemoryStatus::Success: return "Success";
    case pxl::MemoryStatus::ResourceExhaust: return "ResourceExhaust";
    case pxl::MemoryStatus::InvalidRequest: return "InvalidRequest";
    case pxl::MemoryStatus::NotSupported: return "NotSupported";
    default: return "Other";
    }
}

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

int main(int argc, char* argv[])
{
    int deviceId = 0;
    int maxGB = 512;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max") && i + 1 < argc) maxGB = atoi(argv[++i]);
        else { printf("未知参数 %s\n", argv[i]); return 1; }
    }
    auto context = pxl::createContext(deviceId);
    if (!context) { printf("createContext failed\n"); return 1; }

    // ---- 第 1 步:逐个 1 GB ----
    std::vector<void*> slots;
    pxl::MemoryStatus last = pxl::MemoryStatus::Success;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < maxGB; i++)
    {
        void* p = pxl::allocateMemory(deviceId, GB);
        if (!p) { printf("  第 %d 个 1 GB allocateMemory 失败\n", i + 1); break; }
        last = pxl::preloadMemory(p, GB);
        if (last == pxl::MemoryStatus::NotSupported)
        {
            pxl::releaseMemory(p);
            printf("preload 返回 NotSupported:这张卡没有 SSD 层(InfiniteMemory),DRAM 直接映射,\n"
                   "MU 能访问全部卡上 DRAM。可用容量 = daxctl list 里该设备的 size。\n");
            pxl::destroyContext(context);
            return 0;
        }
        if (last != pxl::MemoryStatus::Success)
        {
            pxl::releaseMemory(p);
            break;
        }
        slots.push_back(p);
        if ((i + 1) % 8 == 0) printf("  已 preload %d GB\n", i + 1);
    }
    const int S = (int)slots.size();
    printf("第 1 步:Preload 槽位 = %d 个 1 GB(第 %d 个返回 %s),用时 %.0f ms\n",
           S, S + 1, statusName(last), msSince(t0));
    if (S == 0)
    {
        printf("一个槽位都没有:设备不在 Computing 模式,或 Preload Region 已被别的进程占满\n");
        return 1;
    }

    // ---- 第 2 步:全部释放 ----
    t0 = std::chrono::steady_clock::now();
    int bad = 0;
    for (void* p : slots)
    {
        if (pxl::unloadMemory(p, GB) != pxl::MemoryStatus::Success) bad++;
        pxl::releaseMemory(p);
    }
    printf("第 2 步:unload + release %d 个,失败 %d 个,用时 %.0f ms\n", S, bad, msSince(t0));

    // ---- 第 3 步:最大连续块 ----
    for (int g = S; g >= 1; g--)
    {
        void* p = pxl::allocateMemory(deviceId, (size_t)g * GB);
        if (!p) { printf("  allocateMemory(%d GB) 失败\n", g); continue; }
        t0 = std::chrono::steady_clock::now();
        const pxl::MemoryStatus s = pxl::preloadMemory(p, (size_t)g * GB);
        const double preMs = msSince(t0);
        if (s == pxl::MemoryStatus::Success)
        {
            t0 = std::chrono::steady_clock::now();
            const pxl::MemoryStatus u = pxl::unloadMemory(p, (size_t)g * GB);
            printf("第 3 步:最大连续 preload = %d GB(preload %.0f ms,unload %s %.0f ms)\n",
                   g, preMs, statusName(u), msSince(t0));
            pxl::releaseMemory(p);
            printf("\n结论:MU 可用容量 = %d GB 槽位,单块最大 %d GB\n", S, g);
            pxl::destroyContext(context);
            return 0;
        }
        printf("  %d GB 连续块:%s\n", g, statusName(s));
        pxl::releaseMemory(p);
    }
    printf("第 3 步:连 1 GB 的连续块都失败了\n");
    pxl::destroyContext(context);
    return 1;
}
