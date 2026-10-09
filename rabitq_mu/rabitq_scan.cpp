// rabitq_scan.cpp —— host 驱动:MX1P 上的 RaBitQ 两阶段扫描。
//
// 数据由 prep_rabitq.py 生成(FAISS 编码 + 拆成结构体数组 SoA)。四种 --mode:
//   I  只算第一阶段的两个整数(dot, popcount)       —— 验证位运算路径
//   E  第一阶段 1-bit 距离估计,全部 N 条           —— 验证第一阶段浮点 + 测扫描带宽
//   F  第二阶段多 bit 完整距离,全部 N 条            —— 验证第二阶段浮点
//   T  两阶段 top-k,每 task 各自的堆:rabitq_topk + rabitq_merge
//   G  两阶段 top-k,全局阈值 τ 两遍:rabitq_bounds + rabitq_tau + rabitq_refine + rabitq_merge
// I/E/F 是诊断模式,T/G 是搜索,都只回传 k 个 (距离, id)。G 是该用的那个:T 在 2816 个
// task 下每 task 分到的向量太少,堆填不满,几乎全部被精算(参考模型 FiQA 上 90% vs G 的 0.71%)。排查顺序:I 不对 -> 位运算/布局;I 对 E 不对 ->
// 第一阶段浮点;E 对 F 不对 -> 第二阶段浮点(DRAN 记录过标量浮点循环在 -O3 下算错)。
//
// 内存规则沿用 hamming_scan.cpp:MU 核碰到的一切都在一个 arena 里,preload 一次
// (preloadMemory 以 1 GB 对齐、1 GB 整数倍工作)。全部查询的参数块在 preload 之前
// 一起写进 arena,preload 之后 host 不再写设备内存,只读结果(读前 flush 失效)。
//
// Build: 见 build.sh。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "pxl/pxl.hpp"

namespace
{
constexpr size_t ALIGN_GB = 1ull << 30;
constexpr uint64_t MAGIC_META = 0x31514252ull;   // 'RBQ1',与 prep_rabitq.py 一致
constexpr uint64_t MAGIC_QB = 0x31425152ull;     // 'RQB1'
constexpr uint32_t NONE = 0xFFFFFFFFu;
constexpr uint64_t K_MAX = 64;                   // 与 kernel 一致

inline size_t alignUpGB(size_t x) { return (x + ALIGN_GB - 1) & ~(ALIGN_GB - 1); }

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

bool readExact(const std::string& path, void* dst, size_t bytes)
{
    if (bytes == 0) return true;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { printf("打不开 %s\n", path.c_str()); return false; }
    const size_t got = fread(dst, 1, bytes, f);
    fclose(f);
    if (got != bytes) { printf("%s:读入 %zu B,期望 %zu B\n", path.c_str(), got, bytes); return false; }
    return true;
}

double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

int main(int argc, char* argv[])
{
    const char* data = nullptr;
    const char* mode = "T";
    const char* dump = nullptr;
    int taskCount = 2816;
    int batchSize = 0;          // 0 = Map 默认
    int numSub = 0;             // 0 = createJob() 无参数
    int spread = 0;
    int deviceId = 0;
    int reps = 3;               // 每个查询重复几次,取最小
    long nqLimit = -1;
    int dumpNq = 2;             // I/E/F 模式只写出前几个查询的全部 N 个输出(太大)

    for (int i = 1; i < argc; i++)
    {
        auto nextI = [&](int& idx) { return std::stol(argv[++idx]); };
        if (!strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) taskCount = (int)nextI(i);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) batchSize = (int)nextI(i);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) numSub = (int)nextI(i);
        else if (!strcmp(argv[i], "--spread")) spread = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = (int)nextI(i);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (int)nextI(i);
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) nqLimit = nextI(i);
        else if (!strcmp(argv[i], "--dump-nq") && i + 1 < argc) dumpNq = (int)nextI(i);
        else { printf("未知参数 %s\n", argv[i]); return 1; }
    }
    if (!data) { printf("用法: rabitq_scan --data PREFIX [--mode I|E|F|T|G] [-t 2816] [-b 16] [-s 8] "
                        "[--reps 3] [--nq N] [--dump FILE] [--dump-nq 2]\n"); return 1; }
    const char m = mode[0];
    if (strlen(mode) != 1 || (m != 'I' && m != 'E' && m != 'F' && m != 'T' && m != 'G'))
    { printf("mode 必须是 I、E、F、T 或 G\n"); return 1; }
    const bool topkMode = (m == 'T' || m == 'G');
    if (taskCount <= 0 || reps <= 0) { printf("taskCount、reps 必须 > 0\n"); return 1; }

    // ---- 读元数据 ----
    const std::string P(data);
    uint64_t meta[8];
    if (!readExact(P + ".meta", meta, sizeof(meta))) return 1;
    if (meta[0] != MAGIC_META) { printf("%s.meta 不是 prep_rabitq.py 生成的文件\n", data); return 1; }
    const uint64_t N = meta[1], d = meta[2], words = meta[3], nb = meta[4], exBits = meta[5],
                   exBytes = meta[6];
    uint64_t qh[4];
    {
        FILE* f = fopen((P + ".qblocks").c_str(), "rb");
        if (!f || fread(qh, 1, sizeof(qh), f) != sizeof(qh)) { printf("读不了 qblocks 头\n"); return 1; }
        fclose(f);
    }
    if (qh[0] != MAGIC_QB) { printf("%s.qblocks 头不对\n", data); return 1; }
    uint64_t nq = qh[1];
    const uint64_t qbBytes = qh[2], k = qh[3];
    if (nqLimit > 0 && (uint64_t)nqLimit < nq) nq = (uint64_t)nqLimit;
    if (k == 0 || k > K_MAX) { printf("k=%llu 超出 1..%llu\n", (unsigned long long)k,
                                      (unsigned long long)K_MAX); return 1; }
    if ((m == 'F' || m == 'G') && exBits == 0) { printf("nb=1 没有第二阶段,F/G 模式无意义\n"); return 1; }

    const size_t s1Bytes = words * 8 + 12;            // 第一阶段每向量读的字节
    const size_t s2Bytes = exBytes + 8;               // 第二阶段每向量读的字节
    printf("RaBitQ scan on device %d  [mode %c]\n", deviceId, m);
    printf("  N=%llu d=%llu nb=%llu (exBits=%llu) k=%llu  queries=%llu  tasks=%d batch=%s numSub=%s\n",
           (unsigned long long)N, (unsigned long long)d, (unsigned long long)nb,
           (unsigned long long)exBits, (unsigned long long)k, (unsigned long long)nq, taskCount,
           batchSize ? std::to_string(batchSize).c_str() : "default",
           numSub ? std::to_string(numSub).c_str() : "default");
    printf("  第一阶段 %zu B/向量 (%.3f GB 全库),第二阶段 %zu B/向量\n",
           s1Bytes, N * s1Bytes / 1e9, s2Bytes);

    auto context = pxl::createContext(deviceId);
    if (!context) { printf("createContext failed\n"); return 1; }

    // ---- 单 arena ----
    size_t off = 0;
    auto reserve = [&](size_t bytes, size_t align = 4096) {
        off = (off + align - 1) & ~(align - 1);
        const size_t here = off;
        off += std::max<size_t>(bytes, 64);           // 空区也占位,指针永远合法
        return here;
    };
    const size_t offSigns = reserve(N * words * 8);
    const size_t offFac1 = reserve(N * 12);
    const size_t offEx = reserve(N * exBytes);
    const size_t offFacx = reserve(N * 8);
    const size_t offQb = reserve(nq * qbBytes);
    const size_t outBytesFull = (m == 'I') ? N * 8 : N * 4;
    const size_t offOut = topkMode ? 0 : reserve(outBytesFull);
    const size_t offRes = topkMode ? reserve((size_t)taskCount * k * 8) : 0;
    const size_t offStats = topkMode ? reserve((size_t)taskCount * 8) : 0;
    const size_t offFinal = topkMode ? reserve(k * 8) : 0;
    // G 模式:每条的下界、每 task 的 k 个上界、全局 τ
    const size_t offLB = (m == 'G') ? reserve(N * 4) : 0;
    const size_t offUB = (m == 'G') ? reserve((size_t)taskCount * k * 4) : 0;
    const size_t offTau = (m == 'G') ? reserve(64) : 0;
    const size_t arenaBytes = alignUpGB(off);
    printf("  arena = %.3f GB used -> %.0f GB preloaded\n", off / 1e9, arenaBytes / 1e9);

    auto* arena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, arenaBytes));
    if (!arena) { printf("allocateMemory(%.0f GB) failed\n", arenaBytes / 1e9); return 1; }
    auto* signs = reinterpret_cast<uint64_t*>(arena + offSigns);
    auto* fac1 = reinterpret_cast<float*>(arena + offFac1);
    auto* excode = reinterpret_cast<uint8_t*>(arena + offEx);
    auto* facx = reinterpret_cast<float*>(arena + offFacx);
    auto* qblocks = arena + offQb;
    void* out = topkMode ? nullptr : (void*)(arena + offOut);
    auto* res = reinterpret_cast<uint32_t*>(arena + offRes);
    auto* stats = reinterpret_cast<uint32_t*>(arena + offStats);
    auto* fin = reinterpret_cast<uint32_t*>(arena + offFinal);
    auto* lbArr = reinterpret_cast<float*>(arena + offLB);
    auto* ubArr = reinterpret_cast<float*>(arena + offUB);
    auto* tauBuf = reinterpret_cast<float*>(arena + offTau);

    // ---- 填数据(全部在 preload 之前)----
    auto tFill = std::chrono::steady_clock::now();
    if (!readExact(P + ".signs", signs, N * words * 8) || !readExact(P + ".fac1", fac1, N * 12) ||
        !readExact(P + ".excode", excode, N * exBytes) || !readExact(P + ".facx", facx, N * 8))
        return 1;
    {
        FILE* f = fopen((P + ".qblocks").c_str(), "rb");
        if (!f) { printf("打不开 %s.qblocks\n", data); return 1; }
        fseek(f, 32, SEEK_SET);
        const size_t want = nq * qbBytes;
        const size_t got = fread(qblocks, 1, want, f);
        fclose(f);
        if (got != want) { printf("参数块读入 %zu B,期望 %zu B\n", got, want); return 1; }
    }
    printf("  fill: %.1f ms\n", msSince(tFill));

    // 把主机的写发布到设备内存 —— 少这一步,MU 核读到旧数据,而主机自己读命中缓存、
    // 看起来完全正常(hamming_mu 的教训)。
    auto tFlush = std::chrono::steady_clock::now();
    pxl::flushHostCache(arena, arenaBytes);
    const double flushMs = msSince(tFlush);
    auto tPre = std::chrono::steady_clock::now();
    if (pxl::preloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
    {
        printf("preloadMemory failed —— 设备在 Computing 模式吗?1 GB 槽位够吗?\n");
        pxl::releaseMemory(arena);
        return 1;
    }
    const double preMs = msSince(tPre);
    printf("  flushHostCache %.1f ms, preload %.1f ms\n", flushMs, preMs);

    auto module = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto job = numSub > 0 ? context->createJob(numSub) : context->createJob();
    if (job->load(module) != pxl::Result::Success) { printf("job load failed\n"); return 1; }

    // 第一个(或唯一一个)每-task kernel:I/E/F/T 各自的 kernel;G 是 bounds
    const char* fname = (m == 'I') ? "rabitq_ints" : (m == 'E') ? "rabitq_est1"
                      : (m == 'F') ? "rabitq_full" : (m == 'T') ? "rabitq_topk" : "rabitq_bounds";
    auto* func = module->createFunction(fname);
    auto exec = job->buildMap(func, taskCount);
    if (batchSize > 0) exec->setBatchSize(batchSize);
    if (spread) exec->setLocalityMode(pxl::LocalityMode::SpreadMode);
    auto* mergeFunc = topkMode ? module->createFunction("rabitq_merge") : nullptr;
    auto mergeExec = topkMode ? job->buildMap(mergeFunc, 1) : nullptr;
    // G 模式另外两个:单 task 求 τ,每 task 精算
    auto* tauFunc = (m == 'G') ? module->createFunction("rabitq_tau") : nullptr;
    auto tauExec = (m == 'G') ? job->buildMap(tauFunc, 1) : nullptr;
    auto* refFunc = (m == 'G') ? module->createFunction("rabitq_refine") : nullptr;
    auto refExec = (m == 'G') ? job->buildMap(refFunc, taskCount) : nullptr;
    if (m == 'G')
    {
        if (batchSize > 0) refExec->setBatchSize(batchSize);
        if (spread) refExec->setLocalityMode(pxl::LocalityMode::SpreadMode);
    }

    // 非 const:hamming_scan.cpp 传给 execute 的都是非 const 指针,盲写时不冒类型推导的险
    auto qptr = [&](uint64_t qi) {
        return reinterpret_cast<uint64_t*>(qblocks + qi * qbBytes);
    };
    // 一次执行;返回是否成功。
    //   scanMs  :第一个每-task kernel(I/E/F 的全部;T 的扫描+精算;G 的第一遍 bounds)
    //   mergeMs :其后的全部(T:归并;G:τ + 第二遍精算 + 归并)
    auto runOne = [&](uint64_t qi, double& scanMs, double& mergeMs) -> bool {
        uint64_t* qb = qptr(qi);
        auto t0 = std::chrono::steady_clock::now();
        pxl::Result r;
        if (m == 'I') r = exec->execute(signs, qb, N, (uint32_t*)out);
        else if (m == 'E') r = exec->execute(signs, fac1, qb, N, (float*)out);
        else if (m == 'F') r = exec->execute(signs, excode, facx, qb, N, (float*)out);
        else if (m == 'T') r = exec->execute(signs, fac1, excode, facx, qb, N, res, stats);
        else r = exec->execute(signs, fac1, qb, N, lbArr, ubArr);
        if (r != pxl::Result::Success || exec->synchronize() != pxl::Result::Success) return false;
        scanMs = msSince(t0);
        mergeMs = 0.0;
        if (!topkMode) return true;
        auto t1 = std::chrono::steady_clock::now();
        if (m == 'G')
        {
            if (tauExec->execute(ubArr, (uint64_t)taskCount, k, tauBuf) != pxl::Result::Success ||
                tauExec->synchronize() != pxl::Result::Success)
                return false;
            if (refExec->execute(signs, excode, facx, qb, lbArr, tauBuf, N, res, stats) != pxl::Result::Success ||
                refExec->synchronize() != pxl::Result::Success)
                return false;
        }
        if (mergeExec->execute(res, (uint64_t)taskCount, k, fin) != pxl::Result::Success ||
            mergeExec->synchronize() != pxl::Result::Success)
            return false;
        mergeMs = msSince(t1);
        return true;
    };

    // warmup —— 返回值必须检查:静默失败的 warmup 会让后面的计时建立在"kernel 没跑"上,
    // 症状是数字好得离谱(hamming_mu 的教训)。
    {
        double a, b;
        if (!runOne(0, a, b)) { printf("warmup failed\n"); return 1; }
    }

    FILE* df = dump ? fopen(dump, "wb") : nullptr;
    if (dump && !df) { printf("打不开 %s\n", dump); return 1; }

    std::vector<double> scanT, mergeT, totalT;
    std::vector<double> refinedPerQ;
    uint64_t badProcessed = 0;
    for (uint64_t qi = 0; qi < nq; qi++)
    {
        double bestScan = 1e30, bestMerge = 1e30, bestTotal = 1e30;
        for (int r = 0; r < reps; r++)
        {
            double s = 0, mm = 0;
            if (!runOne(qi, s, mm)) { printf("query %llu execute failed\n", (unsigned long long)qi); return 1; }
            bestScan = std::min(bestScan, s);
            bestMerge = std::min(bestMerge, mm);
            bestTotal = std::min(bestTotal, s + mm);       // 同一次运行的和,不是两个最小值相加
        }
        scanT.push_back(bestScan);
        mergeT.push_back(bestMerge);
        totalT.push_back(bestTotal);

        if (topkMode)
        {
            pxl::flushHostCache(stats, (size_t)taskCount * 8);
            pxl::flushHostCache(fin, k * 8);
            uint64_t refined = 0, processed = 0;
            for (int t = 0; t < taskCount; t++) { refined += stats[2 * t]; processed += stats[2 * t + 1]; }
            if (processed != N) badProcessed++;
            refinedPerQ.push_back((double)refined);
            // 设备给出的 k 个顺序不定,按 (距离, id) 排好再写出
            std::vector<std::pair<float, uint32_t>> v;
            for (uint64_t j = 0; j < k; j++)
            {
                float dv;
                std::memcpy(&dv, &fin[2 * j], 4);
                v.emplace_back(dv, fin[2 * j + 1]);
            }
            std::sort(v.begin(), v.end());
            if (df)
            {
                for (auto& p : v) { fwrite(&p.first, 4, 1, df); fwrite(&p.second, 4, 1, df); }
                fwrite(&refined, 8, 1, df);
                fwrite(&processed, 8, 1, df);
            }
            if (qi < 2)
                printf("  query %llu: top-1 id %u dist %.6f, 精算 %llu / %llu\n", (unsigned long long)qi,
                       v[0].second, v[0].first, (unsigned long long)refined, (unsigned long long)processed);
        }
        else if (df && (int)qi < dumpNq)
        {
            pxl::flushHostCache(out, outBytesFull);       // 读结果前 flush(失效方向)
            fwrite(out, 1, outBytesFull, df);
        }
    }
    if (df) fclose(df);

    // ---- 报告 ----
    const double medScan = median(scanT), minScan = *std::min_element(scanT.begin(), scanT.end());
    const double scanBytes = (m == 'I') ? (double)N * words * 8
                           : (m == 'F') ? (double)N * (words * 8 + exBytes + 8)
                           : (double)N * s1Bytes;
    printf("\n--- 稳态,每查询(%llu 个查询,每个取 %d 次中的最小值)---\n", (unsigned long long)nq, reps);
    printf("  %s 中位 %.3f ms   最小 %.3f ms\n",
           m == 'G' ? "第一遍 bounds" : "扫描 kernel  ", medScan, minScan);
    printf("  扫描读入      %.3f GB  ->  中位有效带宽 %.1f GB/s(上限 268)\n",
           scanBytes / 1e9, scanBytes / 1e9 / (medScan / 1e3));
    if (topkMode)
    {
        const double medRef = median(refinedPerQ);
        printf("  %s 中位 %.3f ms\n", m == 'G' ? "τ+第二遍+归并" : "归并 kernel  ", median(mergeT));
        printf("  合计          中位 %.3f ms\n", median(totalT));
        if (m == 'G')
            printf("  第二遍读入    %.3f GB 下界(4 B/条)+ 被精算向量的符号码与附加码\n", N * 4 / 1e9);
        printf("  精算          中位 %.0f 个/查询 (%.3f%% of N),第二阶段读 %.1f KB/查询\n",
               medRef, 100.0 * medRef / N, medRef * s2Bytes / 1e3);
        printf("  链路回传      %llu B/查询(k 个 (距离, id));设备内实扫 %.3f GB\n",
               (unsigned long long)(k * 8), scanBytes / 1e9);
        if (badProcessed)
            printf("  !! %llu 个查询的各 task 处理数之和 != N —— 分块或 kernel 有问题\n",
                   (unsigned long long)badProcessed);
    }
    printf("\n--- 一次性初始化 ---\n  flushHostCache %.1f ms, preloadMemory %.1f ms\n", flushMs, preMs);
    if (dump) printf("\n  输出已写出 -> %s(用 verify_rabitq.py 校验)\n", dump);

    if (pxl::unloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
        printf("warning: unloadMemory failed\n");
    pxl::releaseMemory(arena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
