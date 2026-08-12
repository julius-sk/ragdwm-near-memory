// hamming_scan.cpp — host 驱动:MX1P 上的 Hamming 扫描与 top-k。
//
// 三组对比中的 B 组(本文件当前实现):设备算距离,回传全部 N 个距离。
// 内存:MU 核touch 的一切都必须在 Preload Region,而 preloadMemory 以
// 1 GB 对齐 + 1 GB 整数倍工作。所以只分配一个 arena,preload 一次,
// 所有 buffer 按偏移切入 —— 单独 preload 一个 32 B 的 query 会烧掉整个 1 GB 槽位。
//
// Build: 见 build.sh。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pxl/pxl.hpp"

namespace
{
constexpr size_t ALIGN_GB = 1ull << 30;
constexpr int WORDS = 4;                 // 每签名 4 x uint64 = 256 位
constexpr size_t SIG_BYTES = WORDS * 8;  // 32 B

inline size_t alignUpGB(size_t x) { return (x + ALIGN_GB - 1) & ~(ALIGN_GB - 1); }

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// xorshift64* —— 与 verify_host.py 无关,仅用于设备侧自生成数据;
// 需要与 numpy 比对时用 --load 指定文件。
struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 88172645463325252ull) {}
    uint64_t next()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
};
}  // namespace

int main(int argc, char* argv[])
{
    size_t numSigs = 1000000;
    int topK = 500;
    int taskCount = 2816;
    int numSub = 0;        // 0 = createJob() 无参数
    int batchSize = 0;     // 0 = 保持 Map 默认(16)
    int spread = 0;
    int deviceId = 0;
    int reps = 5;
    uint64_t seed = 12345;
    const char* mode = "B";
    const char* dumpDists = nullptr;
    const char* loadFile = nullptr;

    for (int i = 1; i < argc; i++)
    {
        auto next = [&](int& idx) { return std::stoll(argv[++idx]); };
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numSigs = (size_t)next(i);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) topK = (int)next(i);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) taskCount = (int)next(i);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) batchSize = (int)next(i);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) numSub = (int)next(i);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (int)next(i);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)next(i);
        else if (!strcmp(argv[i], "--spread")) spread = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = (int)next(i);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--dump-dists") && i + 1 < argc) dumpDists = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) loadFile = argv[++i];
    }

    if (strcmp(mode, "B") != 0)
    {
        printf("mode %s 尚未实现(本任务只做 B)\n", mode);
        return 1;
    }

    const size_t sigBytes = numSigs * SIG_BYTES;
    printf("Hamming scan on device %d  [mode %s]\n", deviceId, mode);
    printf("  N=%zu  k=%d  tasks=%d  batchSize=%s  numSub=%s\n",
           numSigs, topK, taskCount,
           batchSize ? std::to_string(batchSize).c_str() : "default(16)",
           numSub ? std::to_string(numSub).c_str() : "default");
    printf("  signatures = %.3f GB (%zu B each)\n", sigBytes / 1e9, SIG_BYTES);

    auto context = pxl::createContext(deviceId);
    if (!context) { printf("createContext failed\n"); return 1; }

    // ---- 单 arena:MU 核 touch 的一切都在这里 ----
    size_t off = 0;
    auto reserve = [&](size_t bytes, size_t align = 4096) {
        off = (off + align - 1) & ~(align - 1);
        const size_t here = off;
        off += bytes;
        return here;
    };
    const size_t offSig   = reserve(sigBytes);
    const size_t offQuery = reserve(SIG_BYTES);
    const size_t offDists = reserve(numSigs * sizeof(int32_t));
    const size_t arenaBytes = alignUpGB(off);
    printf("  arena = %.3f GB used -> %.0f GB preloaded (%zu x 1 GB slots)\n",
           off / 1e9, arenaBytes / 1e9, arenaBytes / ALIGN_GB);

    auto* arena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, arenaBytes));
    if (!arena) { printf("allocateMemory(%.0f GB) failed\n", arenaBytes / 1e9); return 1; }

    auto* sigs  = reinterpret_cast<uint64_t*>(arena + offSig);
    auto* query = reinterpret_cast<uint64_t*>(arena + offQuery);
    auto* dists = reinterpret_cast<int32_t*>(arena + offDists);

    // ---- 填数据 ----
    auto tFill = std::chrono::steady_clock::now();
    if (loadFile)
    {
        FILE* f = fopen(loadFile, "rb");
        if (!f) { printf("打不开 %s\n", loadFile); return 1; }
        const size_t got = fread(sigs, 1, sigBytes, f);
        fclose(f);
        if (got != sigBytes)
        {
            printf("读入 %zu B,期望 %zu B —— N 与文件不匹配\n", got, sigBytes);
            return 1;
        }
        std::string qf = std::string(loadFile) + ".query";
        FILE* g = fopen(qf.c_str(), "rb");
        if (!g) { printf("打不开 %s\n", qf.c_str()); return 1; }
        if (fread(query, 1, SIG_BYTES, g) != SIG_BYTES) { printf("query 读取失败\n"); return 1; }
        fclose(g);
        printf("  loaded from %s\n", loadFile);
    }
    else
    {
        Rng rng(seed);
        for (size_t i = 0; i < numSigs * WORDS; i++) sigs[i] = rng.next();
        for (int w = 0; w < WORDS; w++) query[w] = rng.next();
    }
    printf("  fill: %.1f ms\n", msSince(tFill));

    // 把主机的写发布到设备内存。少这一步,MU 核读到的是设备 DRAM 里的旧数据,
    // 而主机自己读命中自己的缓存、看起来完全正常 —— 症状是结果错且每次不同。
    auto tFlush = std::chrono::steady_clock::now();
    pxl::flushHostCache(arena, arenaBytes);
    const double flushMs = msSince(tFlush);
    printf("  flushHostCache(%.2f GB): %.1f ms\n", arenaBytes / 1e9, flushMs);

    auto tPre = std::chrono::steady_clock::now();
    if (pxl::preloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
    {
        printf("preloadMemory failed —— 设备在 Computing 模式吗?1 GB 槽位够吗?\n");
        pxl::releaseMemory(arena);
        return 1;
    }
    const double preMs = msSince(tPre);
    printf("  preload arena: %.1f ms\n", preMs);

    // ---- 加载 MU 模块 ----
    auto module = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto job = numSub > 0 ? context->createJob(numSub) : context->createJob();
    if (job->load(module) != pxl::Result::Success) { printf("job load failed\n"); return 1; }

    auto* scanFunc = module->createFunction("hamming_scan_dists");
    auto scanExec = job->buildMap(scanFunc, taskCount);
    if (batchSize > 0) scanExec->setBatchSize(batchSize);
    if (spread) scanExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

    // ---- warmup 一轮,然后跑 reps 次取最小 ----
    if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
    { printf("warmup execute failed\n"); return 1; }
    if (scanExec->synchronize() != pxl::Result::Success)
    { printf("warmup synchronize failed\n"); return 1; }

    double best = 1e30;
    for (int r = 0; r < reps; r++)
    {
        auto t0 = std::chrono::steady_clock::now();
        if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
        { printf("execute failed\n"); return 1; }
        if (scanExec->synchronize() != pxl::Result::Success)
        { printf("synchronize failed\n"); return 1; }
        const double ms = msSince(t0);
        if (ms < best) best = ms;
        printf("  rep %d: %.3f ms\n", r, ms);
    }

    // 读结果之前必须 flush(失效方向)
    auto tRead = std::chrono::steady_clock::now();
    pxl::flushHostCache(dists, numSigs * sizeof(int32_t));
    const double readMs = msSince(tRead);

    const double scannedGB = sigBytes / 1e9;
    printf("\n--- 稳态(数据已在卡上)---\n");
    printf("  scan (min of %d)        : %8.3f ms\n", reps, best);
    printf("  有效带宽                : %8.1f GB/s   (上限 268)\n", scannedGB / (best / 1e3));
    printf("  结果回传 flush          : %8.3f ms\n", readMs);
    printf("\n--- 一次性初始化(稳态不重复)---\n");
    printf("  flushHostCache          : %8.1f ms\n", flushMs);
    printf("  preloadMemory           : %8.1f ms\n", preMs);
    printf("\n--- 链路流量 / 查询 ---\n");
    printf("  in  : query   %zu B\n", SIG_BYTES);
    printf("  out : dists   %.3f MB   <- B 组把全部距离都搬回来了\n",
           numSigs * sizeof(int32_t) / 1e6);
    printf("  设备内实扫    : %.3f GB\n", scannedGB);

    if (dumpDists)
    {
        FILE* f = fopen(dumpDists, "wb");
        if (f)
        {
            fwrite(dists, sizeof(int32_t), numSigs, f);
            fclose(f);
            printf("\n  距离已写出 -> %s\n", dumpDists);
        }
    }

    if (pxl::unloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
        printf("warning: unloadMemory failed\n");
    pxl::releaseMemory(arena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
