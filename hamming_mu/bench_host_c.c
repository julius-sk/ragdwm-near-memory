/* bench_host_c.c — 多线程主机基线(A 组),OpenMP + 硬件 popcount。
 *
 * 为什么需要它:`bench_host_baseline.py` 用 numpy,而 `np.bitwise_count` 和
 * `argpartition` 都是**单线程** ufunc —— 拿它去和用了 176 个 MU 核的设备比,
 * 是本项目最容易被审稿人攻击的地方。本文件给出一个认真实现的主机基线。
 *
 * 设计原则:**基线要取主机能做到的最好,而不是照搬设备算法。**
 * 照搬设备的"直方图 + 候选"两趟做法会让主机多扫一遍 3.2 GB,等于自缚一手。
 * 这里用的是单遍 + 每线程局部 top-k:
 *
 *   - OpenMP 多线程,静态划分,顺序读
 *   - 位打包 uint64 + `__builtin_popcountll`(x86 上是 POPCNT 指令)
 *   - 每线程维护一个升序的 k 元数组,`d >= worst` 一次比较即拒绝。
 *     k=500 时真实第k名距离约 93,命中率 ~1e-5,所以插入几乎不发生,
 *     内层实际只有 4 次 popcount + 1 次比较。
 *   - 收尾用 257 桶计数排序归并 threads×k 条局部结果(几万条,可忽略)
 *
 * 这样主机侧只扫一遍数据,与设备一致,对比才干净。
 *
 * 编译(不需要 CMake):
 *   gcc -O3 -march=native -fopenmp -o bench_host_c bench_host_c.c
 *
 * 用法:
 *   ./bench_host_c -n 100000000 -k 500 --reps 5
 *   ./bench_host_c -n 100000000 -k 500 --load /tmp/sig100m.bin --dump-ids /tmp/host.ids
 *
 * dump 出的 ids 必须过校验 —— 一个快但错的基线看起来像赢:
 *   python verify_large.py check-topk --sigs /tmp/sig100m.bin --ids /tmp/host.ids -k 500
 */

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WORDS 4                  /* 每签名 4 x uint64 = 256 位 */
#define BITS 256
#define NBUCKETS (BITS + 1)      /* 距离 0..256 */
#define SENTINEL (BITS + 1)      /* 局部 top-k 的初始"无穷远" */

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* xorshift64* —— 与 hamming_scan.cpp 的合成数据同一算法 */
static uint64_t rng_next(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

int main(int argc, char **argv)
{
    size_t n = 10000000;
    int topK = 500;
    int reps = 5;
    uint64_t seed = 12345;
    const char *loadFile = NULL;
    const char *dumpIds = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = (size_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) topK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) loadFile = argv[++i];
        else if (!strcmp(argv[i], "--dump-ids") && i + 1 < argc) dumpIds = argv[++i];
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 1; }
    }
    if (topK <= 0 || (size_t)topK > n) {
        fprintf(stderr, "topK 必须在 1..n 之间(topK=%d n=%zu)\n", topK, n);
        return 1;
    }

    const size_t sigBytes = n * WORDS * sizeof(uint64_t);
    uint64_t *sigs = (uint64_t *)malloc(sigBytes);
    uint64_t q[WORDS];
    if (!sigs) { fprintf(stderr, "malloc %.2f GB failed\n", sigBytes / 1e9); return 1; }

    if (loadFile) {
        FILE *f = fopen(loadFile, "rb");
        if (!f) { fprintf(stderr, "打不开 %s\n", loadFile); return 1; }
        if (fread(sigs, 1, sigBytes, f) != sigBytes) {
            fprintf(stderr, "读入不足 %.2f GB —— n 与文件不匹配?\n", sigBytes / 1e9);
            return 1;
        }
        fclose(f);
        char qpath[4096];
        snprintf(qpath, sizeof qpath, "%s.query", loadFile);
        FILE *g = fopen(qpath, "rb");
        if (!g) { fprintf(stderr, "打不开 %s\n", qpath); return 1; }
        if (fread(q, sizeof(uint64_t), WORDS, g) != WORDS) {
            fprintf(stderr, "query 读取失败\n"); return 1;
        }
        fclose(g);
    } else {
        uint64_t s = seed ? seed : 88172645463325252ull;
        for (size_t i = 0; i < n * WORDS; i++) sigs[i] = rng_next(&s);
        for (int w = 0; w < WORDS; w++) q[w] = rng_next(&s);
    }

    const int nthreads = omp_get_max_threads();
    printf("A 组 host 基线 (C + OpenMP)  N=%zu  k=%d  threads=%d\n", n, topK, nthreads);
    printf("  签名 = %.3f GB (%zu B each)%s\n", sigBytes / 1e9,
           (size_t)(WORDS * sizeof(uint64_t)), loadFile ? "  [loaded]" : "  [synthetic]");

    /* 每线程一段局部 top-k(升序),线程间不共享,无锁 */
    int *locD = (int *)malloc((size_t)nthreads * topK * sizeof(int));
    uint32_t *locI = (uint32_t *)malloc((size_t)nthreads * topK * sizeof(uint32_t));
    uint32_t *ids = (uint32_t *)malloc((size_t)topK * sizeof(uint32_t));
    if (!locD || !locI || !ids) { fprintf(stderr, "malloc failed\n"); return 1; }

    double bestScan = 1e30, bestMerge = 1e30, bestTotal = 1e30;
    int kth = BITS;
    int collected = 0;

    /* warmup 一轮不计时,再 reps 次取最小 —— 与设备侧口径一致 */
    for (int r = -1; r < reps; r++) {
        const double t0 = now_ms();

        /* --- 单遍扫描:每线程局部 top-k,早拒绝 --- */
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            int *ld = locD + (size_t)tid * topK;
            uint32_t *li = locI + (size_t)tid * topK;
            for (int j = 0; j < topK; j++) { ld[j] = SENTINEL; li[j] = 0xFFFFFFFFu; }
            int worst = SENTINEL;                 /* == ld[topK-1] */

#pragma omp for schedule(static)
            for (size_t i = 0; i < n; i++) {
                const uint64_t *p = sigs + i * WORDS;
                int d = 0;
                for (int w = 0; w < WORDS; w++)
                    d += __builtin_popcountll(p[w] ^ q[w]);
                if (d >= worst) continue;         /* 绝大多数在这里被一次比较拒掉 */
                int pos = topK - 1;               /* 升序插入 */
                while (pos > 0 && ld[pos - 1] > d) {
                    ld[pos] = ld[pos - 1];
                    li[pos] = li[pos - 1];
                    pos--;
                }
                ld[pos] = d;
                li[pos] = (uint32_t)i;
                worst = ld[topK - 1];
            }
        }
        const double t1 = now_ms();

        /* --- 归并 threads×topK 条局部结果:257 桶计数排序 --- */
        uint64_t hist[NBUCKETS];
        memset(hist, 0, sizeof hist);
        const size_t nloc = (size_t)nthreads * topK;
        for (size_t j = 0; j < nloc; j++)
            if (locD[j] <= BITS) hist[locD[j]]++;

        kth = BITS;
        uint64_t cum = 0;
        for (int b = 0; b < NBUCKETS; b++) {
            cum += hist[b];
            if (cum >= (uint64_t)topK) { kth = b; break; }
        }
        collected = 0;
        for (size_t j = 0; j < nloc && collected < topK; j++)
            if (locD[j] <= kth) ids[collected++] = locI[j];

        const double t2 = now_ms();

        if (r < 0) continue;                      /* warmup 不计时 */
        const double scan = t1 - t0, merge = t2 - t1, total = t2 - t0;
        if (scan < bestScan) bestScan = scan;
        if (merge < bestMerge) bestMerge = merge;
        if (total < bestTotal) bestTotal = total;
        printf("  rep %d: scan %.3f ms  merge %.3f ms  total %.3f ms\n", r, scan, merge, total);
    }

    if (collected < topK)
        fprintf(stderr, "警告:只收集到 %d/%d 个 id\n", collected, topK);

    const double gb = sigBytes / 1e9;
    printf("\n--- A 组 host 基线(min of %d)---\n", reps);
    printf("  scan (单遍 + 局部top-k) : %8.3f ms   (%.1f GB/s)\n", bestScan, gb / (bestScan / 1e3));
    printf("  merge (%d x %d 条)       : %8.3f ms\n", nthreads, topK, bestMerge);
    printf("  合计每查询              : %8.3f ms   (%.1f GB/s 等效)\n",
           bestTotal, gb / (bestTotal / 1e3));
    printf("  第k名距离               : %d\n", kth);
    printf("  需入主机的数据          : %.3f GB\n", gb);
    printf("  线程数                  : %d\n", nthreads);

    if (dumpIds) {
        FILE *f = fopen(dumpIds, "wb");
        if (f) {
            fwrite(ids, sizeof(uint32_t), (size_t)topK, f);
            fclose(f);
            printf("\n  top-k ids 已写出 -> %s\n", dumpIds);
        }
    }

    free(ids); free(locI); free(locD); free(sigs);
    return 0;
}
