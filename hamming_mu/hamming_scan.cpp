// hamming_scan.cpp — host 驱动:MX1P 上的 Hamming 扫描与 top-k。
//
// 三组对比中的 B、C 组(本文件当前实现,--mode 切换):
// B 组设备算距离、回传全部 N 个距离;C 组设备算距离并在设备侧做计数排序
// top-k,只回传 k 个 id。
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
    const char* dumpIds = nullptr;
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
        else if (!strcmp(argv[i], "--dump-ids") && i + 1 < argc) dumpIds = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) loadFile = argv[++i];
    }

    // host 侧防护:kernel 把 candCap/topK 当 int32_t 收,内部转 unsigned 使用 ——
    // <= 0 会静默 wrap 成天文数字,再往后是未定义行为。宁可在 host 上报错退出。
    if (taskCount <= 0)
    {
        printf("taskCount 必须 > 0 (当前 %d)\n", taskCount);
        return 1;
    }
    if (topK <= 0)
    {
        printf("topK 必须 > 0 (当前 %d)\n", topK);
        return 1;
    }
    if ((size_t)topK > numSigs)
    {
        printf("topK (%d) 大于 numSigs (%zu) —— 请求的近邻数超过了签名总数,这是用户错误\n",
               topK, numSigs);
        return 1;
    }

    const bool modeC = (strcmp(mode, "C") == 0);
    if (!modeC && strcmp(mode, "B") != 0)
    {
        printf("mode 必须是 B 或 C\n");
        return 1;
    }
    // 粗阈值:随机 256 位签名的距离集中在 128 附近,取 118 可覆盖 topK
    // 而不至于收进过多候选。溢出时会自动重跑收紧。
    int coarseThresh = 118;
    int candCap = topK * 4;   // 每 task 候选容量
    if (modeC && candCap <= 0)
    {
        printf("candCap 必须 > 0 (当前 %d,来自 topK*4 溢出?)\n", candCap);
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
    const size_t offHist = modeC ? reserve((size_t)taskCount * 257 * sizeof(uint32_t)) : 0;
    // 候选交错存 (id, dist) -> 每条目 2 个 uint32
    const size_t offCand = modeC ? reserve((size_t)taskCount * candCap * 2 * sizeof(uint32_t)) : 0;
    const size_t offMeta = modeC ? reserve((size_t)taskCount * 2 * sizeof(uint32_t)) : 0;
    // outIds 需要 topK + 3 个槽位:尾部 3 个是 kernel 回传的
    // exactThresh / anyOverflow / collected
    const size_t offIds  = modeC ? reserve((size_t)(topK + 3) * sizeof(uint32_t)) : 0;
    const size_t arenaBytes = alignUpGB(off);
    printf("  arena = %.3f GB used -> %.0f GB preloaded (%zu x 1 GB slots)\n",
           off / 1e9, arenaBytes / 1e9, arenaBytes / ALIGN_GB);

    auto* arena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, arenaBytes));
    if (!arena) { printf("allocateMemory(%.0f GB) failed\n", arenaBytes / 1e9); return 1; }

    auto* sigs  = reinterpret_cast<uint64_t*>(arena + offSig);
    auto* query = reinterpret_cast<uint64_t*>(arena + offQuery);
    auto* dists = reinterpret_cast<int32_t*>(arena + offDists);
    auto* hist = modeC ? reinterpret_cast<uint32_t*>(arena + offHist) : nullptr;
    auto* cand = modeC ? reinterpret_cast<uint32_t*>(arena + offCand) : nullptr;
    auto* meta = modeC ? reinterpret_cast<uint32_t*>(arena + offMeta) : nullptr;
    auto* ids  = modeC ? reinterpret_cast<uint32_t*>(arena + offIds)  : nullptr;

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

    double best = 1e30, bestScan = 1e30, bestMerge = 1e30, readMs = 0.0;
    size_t outBytes = 0;

    if (!modeC)
    {
        auto* scanFunc = module->createFunction("hamming_scan_dists");
        auto scanExec = job->buildMap(scanFunc, taskCount);
        if (batchSize > 0) scanExec->setBatchSize(batchSize);
        if (spread) scanExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

        // warmup —— 返回值必须检查。一次静默失败的 warmup 会让后面所有计时
        // 建立在"kernel 其实没跑起来"的基础上,而计时循环自己是检查了的,
        // 于是错误会以"数字异常好看"的形式出现,最难察觉。
        if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
        { printf("warmup execute failed\n"); return 1; }
        if (scanExec->synchronize() != pxl::Result::Success)
        { printf("warmup synchronize failed\n"); return 1; }

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
        bestScan = best;
        outBytes = numSigs * sizeof(int32_t);
        // 读结果之前必须 flush(失效方向)——单独计时,mode C 的等价读在重试
        // 循环内部,那里的计时没有单一有意义的数值,故只在 mode B 报告本行。
        auto tRead = std::chrono::steady_clock::now();
        pxl::flushHostCache(dists, outBytes);
        readMs = msSince(tRead);
    }
    else
    {
        auto* histFunc  = module->createFunction("hamming_scan_hist");
        auto* mergeFunc = module->createFunction("hamming_merge_topk");
        auto histExec  = job->buildMap(histFunc, taskCount);
        auto mergeExec = job->buildMap(mergeFunc, 1);   // 单 task:归约
        if (batchSize > 0) histExec->setBatchSize(batchSize);
        if (spread) histExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

        auto runOnce = [&](double& scanMs, double& mergeMs) -> bool {
            auto t0 = std::chrono::steady_clock::now();
            if (histExec->execute(sigs, query, (uint64_t)numSigs, (int32_t)coarseThresh,
                                  (int32_t)candCap, hist, cand, meta) != pxl::Result::Success)
                return false;
            if (histExec->synchronize() != pxl::Result::Success) return false;
            scanMs = msSince(t0);

            auto t1 = std::chrono::steady_clock::now();
            if (mergeExec->execute(hist, cand, meta, (int32_t)taskCount, (int32_t)candCap,
                                   (int32_t)topK, dists, ids) != pxl::Result::Success)
                return false;
            if (mergeExec->synchronize() != pxl::Result::Success) return false;
            mergeMs = msSince(t1);
            return true;
        };

        double s = 0, m = 0;
        if (!runOnce(s, m)) { printf("warmup failed\n"); return 1; }   // warmup

        // 重试循环。merge 已按 exactThresh 过滤候选,所以收集到的 id 必然满足
        // d <= 第k名距离 —— **溢出不会产生错误结果**,只可能导致收集不足。
        // 因此重试条件只看数量,overflow 退化为诊断信息。
        // kernel 把状态写在 outIds 尾部三个槽位,读之前必须 flush(失效方向)。
        for (int attempt = 0; attempt < 5; attempt++)
        {
            pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
            const int      exactThresh = (int)ids[topK];
            const uint32_t overflow    = ids[topK + 1];
            const uint32_t collected   = ids[topK + 2];

            if (collected >= (uint32_t)topK)
            {
                if (overflow)
                    printf("  注:发生过候选溢出,但结果仍有效(全部 d <= 精确阈值 %d)\n",
                           exactThresh);
                break;
            }
            if (coarseThresh == exactThresh)
            {
                printf("  已在精确阈值 %d 仍只收集到 %u/%d -> candCap(%d)太小\n",
                       exactThresh, collected, topK, candCap);
                break;                       // 重试无用,需增大 candCap
            }
            // 过紧则放宽、过松则收紧 —— 两个方向都收敛到精确阈值
            printf("  收集不足(%u/%d)-> coarseThresh %d -> 精确阈值 %d 重跑\n",
                   collected, topK, coarseThresh, exactThresh);
            coarseThresh = exactThresh;
            if (!runOnce(s, m)) { printf("rescan failed\n"); return 1; }
        }
        pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
        if (ids[topK + 2] < (uint32_t)topK)
        {
            printf("警告:仅收集到 %u/%d 个 id(candCap=%d 太小,提高它重试)\n",
                   ids[topK + 2], topK, candCap);
        }

        for (int r = 0; r < reps; r++)
        {
            if (!runOnce(s, m)) { printf("execute failed\n"); return 1; }
            if (s < bestScan) bestScan = s;
            if (m < bestMerge) bestMerge = m;
            // headline 总数取"同一 rep 内 scan+merge 之和"的最小值,而不是
            // bestScan+bestMerge(两个可能来自不同 rep 的独立最小值相加)——
            // 后者会让总数低于任何一次实际观测到的单次运行,且与 mode B
            // (取 totals 的 min)不对称。bestScan/bestMerge 仍单独保留、
            // 单独报告,仅作为 informational 行。
            if (s + m < best) best = s + m;
            printf("  rep %d: scan %.3f ms  merge %.3f ms\n", r, s, m);
        }
        // 链路流量只算真正的结果(k 个 id);尾部 3 个状态槽是调试用,
        // 生产实现里不需要,故不计入对外报告的流量。
        outBytes = (size_t)topK * sizeof(uint32_t);
        pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
    }

    const double scannedGB = sigBytes / 1e9;
    printf("\n--- 稳态(数据已在卡上)---\n");
    printf("  scan (min of %d)        : %8.3f ms\n", reps, bestScan);
    if (modeC) printf("  merge/top-k             : %8.3f ms\n", bestMerge);
    printf("  合计每查询              : %8.3f ms\n", best);
    printf("  扫描有效带宽            : %8.1f GB/s   (上限 268)\n",
           scannedGB / (bestScan / 1e3));
    if (!modeC) printf("  结果回传 flush          : %8.3f ms\n", readMs);
    printf("\n--- 一次性初始化(稳态不重复)---\n");
    printf("  flushHostCache          : %8.1f ms\n", flushMs);
    printf("  preloadMemory           : %8.1f ms\n", preMs);
    printf("\n--- 链路流量 / 查询 ---\n");
    printf("  in  : query   %zu B\n", SIG_BYTES);
    printf("  out : %s  %.6f MB\n", modeC ? "top-k ids" : "全部距离", outBytes / 1e6);
    printf("  设备内实扫    : %.3f GB\n", scannedGB);
    printf("  流量压缩比    : %.0fx\n", (double)sigBytes / (double)(outBytes + SIG_BYTES));

    if (dumpDists && !modeC)
    {
        FILE* f = fopen(dumpDists, "wb");
        if (f) { fwrite(dists, sizeof(int32_t), numSigs, f); fclose(f);
                 printf("\n  距离已写出 -> %s\n", dumpDists); }
    }
    if (dumpIds && modeC)
    {
        FILE* f = fopen(dumpIds, "wb");
        if (f) { fwrite(ids, sizeof(uint32_t), topK, f); fclose(f);
                 printf("\n  top-k ids 已写出 -> %s\n", dumpIds); }
    }

    if (pxl::unloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
        printf("warning: unloadMemory failed\n");
    pxl::releaseMemory(arena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
