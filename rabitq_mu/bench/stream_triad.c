/* stream_triad.c —— 主机上的「别的工作」:STREAM triad(a = b + s*c),持续占用主机内存带宽。
 *
 * 用途是干扰实验(interference):近内存方案的一条主张是「检索不占主机」。
 * 只比检索本身的延迟证明不了这一点,要看检索和主机上的其他内存密集任务同时跑时,
 * 双方各慢了多少:
 *   1  只跑 stream_triad                     -> 主机可用带宽基准
 *   2  stream_triad + host_rabitq 同时跑      -> 主机检索与它抢带宽
 *   3  stream_triad + rabitq_scan(卡上)同时跑 -> 卡上检索只占 CXL 链路上的几 KB
 * 每秒打印一次带宽,run_bench.sh 取中位数。
 *
 * 编译:gcc -O3 -march=native -fopenmp -o stream_triad stream_triad.c
 * 运行:OMP_NUM_THREADS=96 ./stream_triad --gb 8 --seconds 30
 */
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    double gb = 8.0, seconds = 30.0;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--gb") && i + 1 < argc) gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 1; }
    }
    const size_t n = (size_t)(gb * 1e9 / 3 / sizeof(double));     /* 三个数组共 gb GB */
    double *a = malloc(n * sizeof(double)), *b = malloc(n * sizeof(double)), *c = malloc(n * sizeof(double));
    if (!a || !b || !c) { fprintf(stderr, "内存不够\n"); return 1; }
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) { a[i] = 0; b[i] = 1; c[i] = 2; }
    printf("stream triad: %.1f GB, %d 线程, %.0f s\n", gb, omp_get_max_threads(), seconds);
    fflush(stdout);
    const double t0 = now_s();
    double tl = t0, bytes = 0;
    while (now_s() - t0 < seconds)
    {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; i++) a[i] = b[i] + 3.0 * c[i];
        bytes += 3.0 * n * sizeof(double);
        const double t = now_s();
        if (t - tl >= 1.0)
        {
            printf("  t=%5.1f s  %.1f GB/s\n", t - t0, bytes / (t - tl) / 1e9);
            fflush(stdout);
            bytes = 0;
            tl = t;
        }
    }
    printf("check %.1f\n", a[n / 2]);
    return 0;
}
