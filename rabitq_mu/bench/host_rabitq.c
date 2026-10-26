/* host_rabitq.c —— 主机 CPU 基线:与设备完全相同的 RaBitQ 两阶段搜索,OpenMP 多线程。
 *
 * 读的是同一份 SoA 文件(encode.py / prep_rabitq.py 的输出)和同一份查询参数块,
 * 算式逐句照抄 mu_kernel/mu_rabitq.cpp(第一阶段整数 + fp32 估计,误差界判定,第二阶段
 * 多 bit 距离),所以两边返回的 top-k 应当一致,可以逐 id 互相校验。
 *
 * 设计原则沿用 hamming_mu/bench_host_c.c:**基线要取主机能做到的最好,而不是照搬设备算法。**
 *   - 设备用全局阈值 τ 两遍扫描(G 模式),是因为 2816 个 task 每个分到的向量太少,
 *     堆填不满。主机只有几百个线程,每个线程分到几百万条,堆很快就满了,
 *     所以主机用 FAISS 原本的单遍做法:每线程一个堆,阈值 = 堆中最大值,随扫随收紧。
 *     这样主机只扫一遍、不写下界数组,是对主机最有利的做法。
 *   - 支持一次处理 B 个查询(--qbatch B):每线程按 1024 条一块扫,一块数据在缓存里时
 *     依次给 B 个查询用。B = 1 是单查询延迟;B 大时是吞吐,对应 GPU 的批处理。
 *
 * 编译:gcc -O3 -march=native -fopenmp -o host_rabitq host_rabitq.c -lm
 * 运行:numactl --interleave=all ./host_rabitq --data PREFIX [--qbatch 1] [--nq 100]
 *                                             [--reps 3] [--dump out.bin]
 * dump 格式与 rabitq_scan 的 T/G 模式相同:每查询 k x (float 距离, uint32 id)
 * + uint64 精算次数 + uint64 处理条数,用 eval.py 计算召回、与设备结果对照。
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define K_MAX 64
#define QB_MAX 8
#define WORDS_MAX 48
#define BLOCK 1024
#define MAGIC_META 0x31514252ull
#define MAGIC_QB 0x31425152ull
#define NONE 0xFFFFFFFFu

typedef struct
{
    uint32_t d, words, qb, exBits, exBytes, k, rsv0, rsv1;
    float c1, c2, c34, qrToC, gError, cb, rsv2, rsv3;
} QHeader;

typedef struct
{
    QHeader h;
    const uint64_t *planes;
    const float *rq;
} Query;

typedef struct
{
    float d[K_MAX];
    uint32_t id[K_MAX];
    float thr;      /* 当前最大值(堆未满时为 inf) */
    uint32_t jmax;
    uint64_t refined;
} Heap;

static double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static void *read_file(const char *path, size_t want)
{
    void *p = NULL;
    if (want == 0) return malloc(64);
    if (posix_memalign(&p, 4096, want)) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", path); return NULL; }
    size_t got = 0;
    while (got < want)
    {
        size_t r = fread((char *)p + got, 1, want - got > (1u << 30) ? (1u << 30) : want - got, f);
        if (r == 0) break;
        got += r;
    }
    fclose(f);
    if (got != want) { fprintf(stderr, "%s:读入 %zu B,期望 %zu B\n", path, got, want); return NULL; }
    return p;
}

static inline void heap_reset(Heap *hp, uint32_t k)
{
    for (uint32_t j = 0; j < k; j++) { hp->d[j] = INFINITY; hp->id[j] = NONE; }
    hp->thr = INFINITY;
    hp->jmax = 0;
    hp->refined = 0;
}

static inline void heap_put(Heap *hp, uint32_t k, float v, uint32_t id)
{
    hp->d[hp->jmax] = v;
    hp->id[hp->jmax] = id;
    uint32_t jm = 0;
    for (uint32_t j = 1; j < k; j++)
        if (hp->d[j] > hp->d[jm]) jm = j;
    hp->jmax = jm;
    hp->thr = hp->d[jm];
}

/* 与 kernel 的 stage1Ints + stage1Est 同一算式、同一运算顺序 */
static inline float stage1_est(const uint64_t *s, const float *f1, const Query *q)
{
    uint64_t dot = 0, pc = 0;
    for (uint32_t w = 0; w < q->h.words; w++)
    {
        const uint64_t x = s[w];
        pc += (uint64_t)__builtin_popcountll(x);
        for (uint32_t j = 0; j < q->h.qb; j++)
            dot += (uint64_t)__builtin_popcountll(q->planes[j * q->h.words + w] & x) << j;
    }
    float fd = 0.0f;
    fd += q->h.c1 * (float)dot;
    fd += q->h.c2 * (float)pc;
    fd -= q->h.c34;
    const float pre = (f1[0] + q->h.qrToC) - 2.0f * f1[1] * fd;
    return pre > 0.0f ? pre : 0.0f;
}

static inline float stage2_full(const uint64_t *s, const uint8_t *ex, const float *fx, const Query *q)
{
    const uint32_t eb = q->h.exBits, mask = (1u << eb) - 1u;
    float acc = 0.0f;
    for (uint32_t i = 0; i < q->h.d; i++)
    {
        const int sb = (int)((s[i >> 6] >> (i & 63u)) & 1ull);
        const uint32_t bp = i * eb, byteIdx = bp >> 3, off = bp & 7u;
        uint32_t raw = ex[byteIdx];
        if (off + eb > 8u) raw |= (uint32_t)ex[byteIdx + 1] << 8;
        const int exv = (int)((raw >> off) & mask);
        acc += q->rq[i] * ((float)((sb << eb) + exv) + q->h.cb);
    }
    const float dist = q->h.qrToC + fx[0] + fx[1] * acc;
    return dist > 0.0f ? dist : 0.0f;
}

int main(int argc, char **argv)
{
    const char *data = NULL, *dump = NULL;
    int qbatch = 1, reps = 3;
    long nqLimit = -1;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--data") && i + 1 < argc) data = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--qbatch") && i + 1 < argc) qbatch = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--nq") && i + 1 < argc) nqLimit = atol(argv[++i]);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 1; }
    }
    if (!data || qbatch < 1 || reps < 1)
    {
        fprintf(stderr, "用法: host_rabitq --data PREFIX [--qbatch 1] [--nq N] [--reps 3] [--dump F]\n");
        return 1;
    }
    char path[4096];
    uint64_t meta[8];
    snprintf(path, sizeof path, "%s.meta", data);
    uint64_t *mp = read_file(path, sizeof meta);
    if (!mp || mp[0] != MAGIC_META) { fprintf(stderr, "%s 不是 RBQ1 元数据\n", path); return 1; }
    memcpy(meta, mp, sizeof meta);
    const uint64_t N = meta[1], d = meta[2], words = meta[3], nb = meta[4], exBits = meta[5],
                   exBytes = meta[6];

    snprintf(path, sizeof path, "%s.qblocks", data);
    FILE *fq = fopen(path, "rb");
    uint64_t qh[4];
    if (!fq || fread(qh, 8, 4, fq) != 4 || qh[0] != MAGIC_QB) { fprintf(stderr, "qblocks 头不对\n"); return 1; }
    uint64_t nq = qh[1];
    const uint64_t qbBytes = qh[2], k = qh[3];
    if (nqLimit > 0 && (uint64_t)nqLimit < nq) nq = (uint64_t)nqLimit;
    if (k == 0 || k > K_MAX) { fprintf(stderr, "k 超出范围\n"); return 1; }
    uint8_t *qraw = malloc(nq * qbBytes);
    if (fread(qraw, 1, nq * qbBytes, fq) != nq * qbBytes) { fprintf(stderr, "qblocks 读不全\n"); return 1; }
    fclose(fq);
    Query *Q = malloc(nq * sizeof(Query));
    for (uint64_t i = 0; i < nq; i++)
    {
        const uint8_t *b = qraw + i * qbBytes;
        memcpy(&Q[i].h, b, sizeof(QHeader));
        Q[i].planes = (const uint64_t *)(b + sizeof(QHeader));
        Q[i].rq = (const float *)(b + sizeof(QHeader) + 8ull * Q[i].h.qb * Q[i].h.words);
        if (Q[i].h.words != words || Q[i].h.qb == 0 || Q[i].h.qb > QB_MAX) { fprintf(stderr, "参数块不匹配\n"); return 1; }
    }

    double t0 = now_ms();
    snprintf(path, sizeof path, "%s.signs", data);
    const uint64_t *signs = read_file(path, N * words * 8);
    snprintf(path, sizeof path, "%s.fac1", data);
    const float *fac1 = read_file(path, N * 12);
    snprintf(path, sizeof path, "%s.excode", data);
    const uint8_t *excode = read_file(path, N * exBytes);
    snprintf(path, sizeof path, "%s.facx", data);
    const float *facx = read_file(path, N * 8);
    if (!signs || !fac1 || !excode || !facx) return 1;
    const int T = omp_get_max_threads();
    const double gb1 = N * (words * 8 + 12) / 1e9, gb2 = N * (exBytes + 8) / 1e9;
    printf("host RaBitQ  N=%llu d=%llu nb=%llu k=%llu  queries=%llu  threads=%d qbatch=%d\n",
           (unsigned long long)N, (unsigned long long)d, (unsigned long long)nb,
           (unsigned long long)k, (unsigned long long)nq, T, qbatch);
    printf("  主机内存中:第一阶段 %.2f GB + 第二阶段 %.2f GB,载入 %.0f ms\n", gb1, gb2, now_ms() - t0);

    /* 每线程每查询一个堆 */
    Heap *heaps = malloc((size_t)T * qbatch * sizeof(Heap));
    FILE *df = dump ? fopen(dump, "wb") : NULL;
    double *lat = malloc(((nq + qbatch - 1) / qbatch) * sizeof(double));
    uint64_t *refinedQ = calloc(nq, sizeof(uint64_t));
    double tAll = 0.0;
    uint64_t nbatch = 0;

    for (uint64_t q0 = 0; q0 < nq; q0 += qbatch, nbatch++)
    {
        const int B = (int)((nq - q0) < (uint64_t)qbatch ? (nq - q0) : (uint64_t)qbatch);
        double best = 1e30;
        for (int r = 0; r < reps; r++)
        {
            const double ts = now_ms();
#pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                Heap *hs = heaps + (size_t)tid * qbatch;
                for (int b = 0; b < B; b++) heap_reset(&hs[b], (uint32_t)k);
                const uint64_t per = (N + T - 1) / T;
                const uint64_t beg = (uint64_t)tid * per > N ? N : (uint64_t)tid * per;
                const uint64_t end = beg + per > N ? N : beg + per;
                for (uint64_t blk = beg; blk < end; blk += BLOCK)
                {
                    const uint64_t be = blk + BLOCK > end ? end : blk + BLOCK;
                    for (int b = 0; b < B; b++)
                    {
                        const Query *q = &Q[q0 + b];
                        Heap *hp = &hs[b];
                        for (uint64_t i = blk; i < be; i++)
                        {
                            const float *f1 = fac1 + 3 * i;
                            const uint64_t *s = signs + i * words;
                            const float est = stage1_est(s, f1, q);
                            float lb = est - f1[2] * q->h.gError;
                            if (lb < 0.0f) lb = 0.0f;
                            if (lb < hp->thr)
                            {
                                hp->refined++;
                                const float full = exBits == 0 ? est
                                    : stage2_full(s, excode + i * exBytes, facx + 2 * i, q);
                                if (full < hp->thr) heap_put(hp, (uint32_t)k, full, (uint32_t)i);
                            }
                        }
                    }
                }
            }
            const double el = now_ms() - ts;
            if (el < best) best = el;
        }
        /* 归并(串行,T x k 条,可忽略)+ 写出 */
        for (int b = 0; b < B; b++)
        {
            float bd[K_MAX];
            uint32_t bi[K_MAX];
            uint64_t refined = 0;
            for (uint64_t j = 0; j < k; j++) { bd[j] = INFINITY; bi[j] = NONE; }
            for (int t = 0; t < T; t++)
            {
                const Heap *hp = &heaps[(size_t)t * qbatch + b];
                refined += hp->refined;
                for (uint64_t e = 0; e < k; e++)
                {
                    if (hp->id[e] == NONE) continue;
                    uint64_t jw = 0;
                    for (uint64_t j = 1; j < k; j++)
                        if (bd[j] > bd[jw] || (bd[j] == bd[jw] && bi[j] > bi[jw])) jw = j;
                    if (hp->d[e] < bd[jw] || (hp->d[e] == bd[jw] && hp->id[e] < bi[jw]))
                    { bd[jw] = hp->d[e]; bi[jw] = hp->id[e]; }
                }
            }
            /* 按 (距离, id) 排好再写,与 rabitq_scan 的 dump 一致 */
            for (uint64_t a1 = 0; a1 < k; a1++)
                for (uint64_t a2 = a1 + 1; a2 < k; a2++)
                    if (bd[a2] < bd[a1] || (bd[a2] == bd[a1] && bi[a2] < bi[a1]))
                    {
                        float tf = bd[a1]; bd[a1] = bd[a2]; bd[a2] = tf;
                        uint32_t ti = bi[a1]; bi[a1] = bi[a2]; bi[a2] = ti;
                    }
            refinedQ[q0 + b] = refined;
            if (df)
            {
                for (uint64_t j = 0; j < k; j++) { fwrite(&bd[j], 4, 1, df); fwrite(&bi[j], 4, 1, df); }
                uint64_t proc = N;
                fwrite(&refined, 8, 1, df);
                fwrite(&proc, 8, 1, df);
            }
            if (q0 + b < 2)
                printf("  query %llu: top-1 id %u dist %.6f, 精算 %llu / %llu\n",
                       (unsigned long long)(q0 + b), bi[0], bd[0], (unsigned long long)refined,
                       (unsigned long long)N);
        }
        lat[nbatch] = best;
        tAll += best;
    }
    if (df) fclose(df);

    /* 中位数 */
    for (uint64_t a1 = 0; a1 < nbatch; a1++)
        for (uint64_t a2 = a1 + 1; a2 < nbatch; a2++)
            if (lat[a2] < lat[a1]) { double t = lat[a1]; lat[a1] = lat[a2]; lat[a2] = t; }
    uint64_t rs = 0;
    for (uint64_t i = 0; i < nq; i++) rs += refinedQ[i];
    const double med = lat[nbatch / 2];
    printf("\n--- 稳态(每批 %d 个查询,每批取 %d 次中的最小值)---\n", qbatch, reps);
    printf("  每批延迟      中位 %.3f ms   最小 %.3f ms\n", med, lat[0]);
    printf("  吞吐          %.1f 查询/s\n", nq / (tAll / 1e3));
    printf("  第一阶段带宽  %.1f GB/s(中位批,数据每批读一遍)\n", gb1 / (med / 1e3));
    printf("  精算          平均 %.0f 个/查询 (%.4f%% of N)\n", (double)rs / nq, 100.0 * rs / nq / N);
    if (dump) printf("  输出 -> %s(用 eval.py 校验)\n", dump);
    return 0;
}
