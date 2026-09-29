/* bench_roofline.c —— 主机侧 roofline 对照,回答"51.8 GB/s 是硬上限还是实现上限"。
 *
 * 为什么需要它。RESULTS.md 记录:主机 C/OpenMP 基线 51.8 GB/s,设备 35.3 GB/s,
 * 设备拿到主机的 68%。但同一份记录里还有一行 objdump 结果:
 *
 *     主机侧只有 4 条标量 POPCNT、0 条 VPOPCNTDQ —— gcc 没有向量化。
 *
 * 而双路 Granite Rapids 的内存带宽在 500 GB/s 量级。一个跑到 51.8 GB/s 的扫描
 * 大约只用了 10% 的可用带宽,也就是说**主机基线自己就没跑满**。如果把它写对,
 * 它可能直接越过设备的天花板,那样近内存这条路就没有意义了。
 *
 * 这个判断不能靠推理,必须测。本文件测四个数:
 *
 *   1. bw_read       纯顺序读 + XOR 累加,不做 popcount。主机内存 roofline。
 *   2. scan_scalar   现有实现的做法:内层同时算距离和维护 top-k。
 *   3. scan_blocked  距离循环与 top-k 循环分离,让编译器能向量化距离循环。
 *   4. (可选) 每个变体离 bw_read 有多远 —— 这才是"实现上限 vs 硬上限"的答案。
 *
 * 为什么用分块而不是手写 intrinsics。scan_scalar 之所以没被向量化,几乎肯定是
 * 因为内层混进了 top-k 的分支。分块版把距离计算变成一段无分支的定长循环,
 * 编译器可以自行向量化成 VPOPCNTDQ,而**正确性由编译器保证,不由我保证** ——
 * 手写一段无法在本机验证的 AVX-512 是这个项目最不该冒的险。
 *
 * 向量化是否真的发生了,不许假定,必须查:
 *     objdump -d bench_roofline | grep -c vpopcnt
 * 计数为 0 就说明分块没生效,这时报告里的 scan_blocked 数字没有意义。
 * 本程序退出前会自己提示这条命令。
 *
 * 编译(至少需要支持 AVX-512 VPOPCNTDQ 的编译器与 CPU;Granite Rapids 有):
 *     gcc -O3 -march=native -fopenmp -o bench_roofline bench_roofline.c
 * 若 -march=native 未启用 vpopcntdq,显式加:
 *     gcc -O3 -mavx512f -mavx512vpopcntdq -mtune=native -fopenmp -o bench_roofline bench_roofline.c
 *
 * 用法:
 *     ./bench_roofline -n 100000000 -k 500 --reps 5
 *
 * 正确性:三个扫描变体必须返回同一个第 k 名距离和同一个距离集合。一个快但错的
 * 基线看起来就是赢,所以校验不是可选项,不过则直接非零退出。
 */

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define WORDS 4                 /* 每签名 4 x uint64 = 256 位,与设备布局一致 */
#define BITS 256
#define NBUCKETS (BITS + 1)
#define BLOCK 4096              /* 分块大小:距离缓冲区留在 L1/L2 内 */

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

/* xorshift64* —— 与 hamming_scan.cpp 和 bench_host_c.c 的合成数据同一算法,
 * 三者必须生成完全相同的签名,否则跨文件的结果没有可比性。 */
static uint64_t rng_next(uint64_t *s)
{
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    return *s;
}

/* ---------------------------------------------------------------------------
 * 1. 内存 roofline:纯顺序读,每 8 字节一次 XOR 累加。
 *
 * 没有 popcount、没有比较、没有写出。这是任何扫描 kernel 在这台机器上的上限。
 * 累加值最后要被使用,否则 -O3 会把整个循环删掉 —— 这是这类基准最常见的 bug。
 * --------------------------------------------------------------------------- */
static uint64_t bw_read(const uint64_t *sigs, size_t nwords)
{
    uint64_t sink = 0;
#pragma omp parallel reduction(^ : sink)
    {
        uint64_t acc = 0;
#pragma omp for schedule(static)
        for (size_t i = 0; i < nwords; i++) acc ^= sigs[i];
        sink ^= acc;
    }
    return sink;
}

/* ---------------------------------------------------------------------------
 * 2. 标量扫描:距离与 top-k 在同一个内层循环里(现有 bench_host_c.c 的结构)。
 *
 * 每线程一个升序 k 元数组,d >= worst 一次比较即拒绝。k=500 时真实第 k 名距离
 * 约 93,命中率 ~1e-5,插入几乎不发生。
 * --------------------------------------------------------------------------- */
static void topk_insert(int32_t *arr, int k, int32_t d)
{
    if (d >= arr[k - 1]) return;
    int j = k - 1;
    while (j > 0 && arr[j - 1] > d) { arr[j] = arr[j - 1]; j--; }
    arr[j] = d;
}

static void scan_scalar(const uint64_t *sigs, size_t n, const uint64_t *q,
                        int k, int32_t *out_topk)
{
    const int nt = omp_get_max_threads();
    int32_t *local = malloc((size_t)nt * k * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)nt * k; i++) local[i] = BITS + 1;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int32_t *mine = local + (size_t)tid * k;
#pragma omp for schedule(static)
        for (size_t i = 0; i < n; i++)
        {
            const uint64_t *s = sigs + i * WORDS;
            int d = 0;
            for (int w = 0; w < WORDS; w++) d += __builtin_popcountll(s[w] ^ q[w]);
            topk_insert(mine, k, (int32_t)d);
        }
    }

    /* 257 桶计数排序归并 nt*k 条局部结果 —— 几万条,相对扫描可忽略 */
    int32_t hist[NBUCKETS] = {0};
    for (size_t i = 0; i < (size_t)nt * k; i++)
        if (local[i] <= BITS) hist[local[i]]++;
    int got = 0;
    for (int d = 0; d <= BITS && got < k; d++)
        for (int c = 0; c < hist[d] && got < k; c++) out_topk[got++] = d;
    while (got < k) out_topk[got++] = BITS + 1;
    free(local);
}

/* ---------------------------------------------------------------------------
 * 3. 分块扫描:距离循环无分支、定长,编译器可向量化;top-k 在块外做。
 *
 * 这是本文件存在的理由。如果这一版显著快于 scan_scalar,那么 RESULTS.md 里
 * 那个 51.8 GB/s 是**实现上限**而不是硬件上限,设备与主机的差距会被重新定义。
 * --------------------------------------------------------------------------- */
static void scan_blocked(const uint64_t *sigs, size_t n, const uint64_t *q,
                         int k, int32_t *out_topk)
{
    const int nt = omp_get_max_threads();
    int32_t *local = malloc((size_t)nt * k * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)nt * k; i++) local[i] = BITS + 1;

    const uint64_t q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3];

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        int32_t *mine = local + (size_t)tid * k;
        int32_t buf[BLOCK];

#pragma omp for schedule(static)
        for (size_t b = 0; b < (n + BLOCK - 1) / BLOCK; b++)
        {
            const size_t beg = b * BLOCK;
            size_t len = n - beg;
            if (len > BLOCK) len = BLOCK;
            const uint64_t *s = sigs + beg * WORDS;

            /* 无分支定长循环 —— 展开成 4 个独立的 popcount 链,没有依赖、
             * 没有提前退出,是编译器最容易向量化的形状。 */
            for (size_t i = 0; i < len; i++)
                buf[i] = (int32_t)(__builtin_popcountll(s[i * 4 + 0] ^ q0) +
                                   __builtin_popcountll(s[i * 4 + 1] ^ q1) +
                                   __builtin_popcountll(s[i * 4 + 2] ^ q2) +
                                   __builtin_popcountll(s[i * 4 + 3] ^ q3));

            /* top-k 在块外,分支留在这里,不污染上面的循环 */
            const int32_t worst = mine[k - 1];
            for (size_t i = 0; i < len; i++)
                if (buf[i] < worst || mine[k - 1] > buf[i]) topk_insert(mine, k, buf[i]);
        }
    }

    int32_t hist[NBUCKETS] = {0};
    for (size_t i = 0; i < (size_t)nt * k; i++)
        if (local[i] <= BITS) hist[local[i]]++;
    int got = 0;
    for (int d = 0; d <= BITS && got < k; d++)
        for (int c = 0; c < hist[d] && got < k; c++) out_topk[got++] = d;
    while (got < k) out_topk[got++] = BITS + 1;
    free(local);
}

/* --------------------------------------------------------------------------- */

static double gbps(size_t bytes, double ms) { return bytes / 1e9 / (ms / 1e3); }

int main(int argc, char **argv)
{
    size_t n = 100000000;
    int k = 500, reps = 5;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = strtoull(argv[++i], 0, 10);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else { fprintf(stderr, "用法: %s [-n N] [-k K] [--reps R]\n", argv[0]); return 2; }
    }

    const size_t bytes = n * WORDS * sizeof(uint64_t);
    printf("N = %zu 签名,%.3f GB,k = %d,线程 %d,重复 %d 次取最小\n\n",
           n, bytes / 1e9, k, omp_get_max_threads(), reps);

    uint64_t *sigs = aligned_alloc(64, bytes);
    if (!sigs) { fprintf(stderr, "分配 %.1f GB 失败\n", bytes / 1e9); return 1; }

    /* 并行填充:同时也是 first-touch,让每页落在使用它的线程的 NUMA 节点上。
     * 单线程填充会把所有页放在一个 socket 上,双路机器上会凭空损失一半带宽,
     * 而且这种损失完全不显眼 —— 它只会让"主机很慢"这个结论看起来成立。 */
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++)
    {
        uint64_t st = 0x9E3779B97F4A7C15ull ^ (i * 0xBF58476D1CE4E5B9ull);
        if (!st) st = 1;
        for (int w = 0; w < WORDS; w++) sigs[i * WORDS + w] = rng_next(&st);
    }
    uint64_t q[WORDS];
    { uint64_t st = 0xDEADBEEFCAFEBABEull;
      for (int w = 0; w < WORDS; w++) q[w] = rng_next(&st); }

    /* ---- 1. 内存 roofline ---- */
    uint64_t sink = bw_read(sigs, n * WORDS);      /* warmup */
    double best_bw = 1e30;
    for (int r = 0; r < reps; r++)
    {
        double t0 = now_ms();
        sink ^= bw_read(sigs, n * WORDS);
        double ms = now_ms() - t0;
        if (ms < best_bw) best_bw = ms;
    }

    /* ---- 2 & 3. 两个扫描变体 ---- */
    int32_t *tk_scalar = malloc(k * sizeof(int32_t));
    int32_t *tk_block  = malloc(k * sizeof(int32_t));

    scan_scalar(sigs, n, q, k, tk_scalar);         /* warmup */
    double best_sc = 1e30;
    for (int r = 0; r < reps; r++)
    {
        double t0 = now_ms();
        scan_scalar(sigs, n, q, k, tk_scalar);
        double ms = now_ms() - t0;
        if (ms < best_sc) best_sc = ms;
    }

    scan_blocked(sigs, n, q, k, tk_block);         /* warmup */
    double best_bl = 1e30;
    for (int r = 0; r < reps; r++)
    {
        double t0 = now_ms();
        scan_blocked(sigs, n, q, k, tk_block);
        double ms = now_ms() - t0;
        if (ms < best_bl) best_bl = ms;
    }

    /* ---- 正确性:两个扫描变体必须给出完全相同的 top-k 距离多重集 ---- */
    int bad = 0;
    for (int i = 0; i < k; i++) if (tk_scalar[i] != tk_block[i]) bad++;
    if (bad)
    {
        printf("*** 校验失败:scan_scalar 与 scan_blocked 的 top-k 有 %d 处不同 ***\n", bad);
        printf("    第 k 名距离 scalar=%d blocked=%d\n", tk_scalar[k - 1], tk_block[k - 1]);
        printf("    快但错的基线看起来就是赢,所以这里不继续报告性能数字。\n");
        return 1;
    }

    printf("校验通过:两个扫描变体的 top-k 距离完全一致,第 k 名 = %d\n\n", tk_scalar[k - 1]);
    printf("%-34s %12s %14s %12s\n", "", "时间 (ms)", "有效带宽 GB/s", "占 roofline");
    printf("%-34s %12.3f %14.1f %11s\n", "1. 纯顺序读 (内存 roofline)",
           best_bw, gbps(bytes, best_bw), "100%");
    printf("%-34s %12.3f %14.1f %10.1f%%\n", "2. 标量扫描 (现有实现的结构)",
           best_sc, gbps(bytes, best_sc), 100.0 * best_bw / best_sc);
    printf("%-34s %12.3f %14.1f %10.1f%%\n", "3. 分块扫描 (可向量化)",
           best_bl, gbps(bytes, best_bl), 100.0 * best_bw / best_bl);
    printf("\n分块相对标量:%.2fx\n", best_sc / best_bl);

    printf("\n--- 必须自己确认的两件事 ---\n");
    printf("1. 向量化是否真的发生:\n");
    printf("     objdump -d %s | grep -c vpopcnt\n", argv[0]);
    printf("   为 0 表示编译器没有向量化,上面第 3 行的数字不能用作结论。\n");
    printf("2. 与设备对照时,N 必须相同(RESULTS.md 的设备数字取自 N=100000000)。\n");
    printf("\n(防优化 sink = %llu)\n", (unsigned long long)sink);

    free(sigs); free(tk_scalar); free(tk_block);
    return 0;
}
