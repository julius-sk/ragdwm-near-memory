# bench —— 在 solab-h1 上用公开数据集做真实负载测试

目标是用别人论文里的数据集测 RaBitQ 两阶段搜索,这样数字能和他们直接对照。同一份码分别在三处跑:

| 路线 | 在哪算 | 码放在哪 |
|---|---|---|
| 卡 | MX1P 的 MU 核(`../rabitq_scan`,G 模式) | 卡上 DRAM 的 Preload Region |
| 主机 C | 主机 CPU,同一算法,OpenMP(`host_rabitq`) | 主机内存 |
| FAISS | 主机 CPU,公开库 `IndexRaBitQ`(`faiss_host.py`) | 主机内存 |

三条路线读同一份 SoA 文件和同一份查询参数块。卡和主机 C 的 top-k 会逐 id 互相对照。

## 数据集

都是相关论文实际用过的。下载地址在 2026-10-23 核对过,详见 `RAGDWM/research_notes/h1_workload_datasets.md`。

| 名称 | 规模 | 谁用过 | GT | 报什么 |
|---|---|---|---|---|
| `gist1m` | 100 万 × 960 | RaBitQ、Extended RaBitQ、GPU IVF-RaBitQ、DiskANN | 自己算,并与官方 ivecs 核对 | recall@10 |
| `openai3072` / `openai1536` | 99.9 万 × 3072 / 1536 | Extended RaBitQ、GPU IVF-RaBitQ | 自己算(留出 1,000 条当查询) | recall@10 |
| `deep10m` / `deep100m` / `deep1b` | 96 维 | CXL-ANNS、DiskANN、SPANN、REIS、big-ann | 官方 | recall@10 |
| `beir_nq` / `beir_hotpotqa` / `beir_msmarco` | 268 万 / 523 万 / 884 万 × 1024(Cohere v3) | REIS(NQ、HotpotQA) | 自己算 + 人工标注 qrels | recall@10、**nDCG@10** |

`deep10m` 和 `deep100m` 只下载 `base.1B.fbin` 的前一段,分别约 3.8 GB 和 38 GB。`deep1b` 要 384 GB 的盘。

## 已在开发机上验证(2026-10-23)

- 流式编码(`encode.py`):先设好中心再 `sa_encode`,得到的码与 FAISS 整库 `train + add` 逐字节相同。
- 合成的各向异性数据上,模拟的卡输出与 FAISS、主机 C 的 top-10 id 集合 100% 一致。主机 C 的批量模式(64 个查询一批)与单查询模式结果也一致。
- 随机旋转在这类数据上把 4 bit 的 recall@10 从 0.50 提到 0.83,所以 `encode.py` 默认开启。
- BEIR SciFact(Cohere v3):精确搜索的 nDCG@10 = 0.718,与 MTEB 公开值一致。2 bit / 4 bit 的 nDCG@10 为 0.720 / 0.721,与精确搜索持平。
- 重排(`rescore.py`):卡返回 64 个候选,读原始向量重算后,recall@10 从 0.82 升到 1.00。
- `host_rabitq.c`、`preload_probe.cpp` 用 zig 做了 Linux 目标的编译检查;kernel 的维度上限从 2048 提到 3072(为了 OpenAI 3072 维)。

**没在硬件上跑过的:** 卡上的一切、`preload_probe`、`fetch.sh` 的大文件下载。

## 在 h1 上怎么跑

先读 `RAGDWM/xcena/xcena-mx1p-playbook.md`,并确认卡没人在用。全部步骤都在 tmux 里跑。

### 第 0 步:拉代码、构建、量清楚机器

```
1  tmux new -s rbq
2  cd ~/engramme && git pull origin master
3  cd rabitq_mu/bench
4  pip install numpy faiss-cpu numba h5py pyarrow
5  bash build_bench.sh
6  bash h1_probe.sh > probe.txt 2>&1
```

把 `probe.txt` 发回来。里面最重要的一行是「最大连续 preload」,后面用 `PRELOAD_GB` 传给 `run_bench.sh`。

如果 `Computable` 不是 Yes,按 XCENA 2026-10-26 的说明执行 `sudo xcena_cli imc enable-compute 0`,然后重启。**不要刷固件。**

### 第 1 步:卡上正确性(沿用原来的脚本)

```
7  cd ~/engramme/rabitq_mu
8  bash run_rabitq.sh
```

I → E → F → T → G 全部通过后,再继续往下。

### 第 2 步:按从小到大的顺序跑数据集

每个数据集执行两条命令:先下载,再跑。`PRELOAD_GB` 用第 0 步量出来的数。

```
9   cd ~/engramme/rabitq_mu/bench
10  export PRELOAD_GB=<第0步的数>
11  bash fetch.sh gist1m
12  bash run_bench.sh gist1m
13  bash fetch.sh beir-nq
14  bash run_bench.sh beir_nq
15  bash fetch.sh beir-hotpotqa
16  bash run_bench.sh beir_hotpotqa
17  bash fetch.sh openai
18  bash run_bench.sh openai1536
19  bash run_bench.sh openai3072
20  bash fetch.sh deep10m
21  bash run_bench.sh deep10m
22  bash fetch.sh deep100m
23  bash run_bench.sh deep100m
```

每次跑完,把 `results/<数据集>_<时间>/summary.txt` 发回来。证书报错时在命令前加 `CURL_K=1`。

### 第 3 步:干扰实验

在卡上检索和主机检索两种情况下,分别与主机上的内存密集任务并发运行,看双方各自慢了多少。

```
24  bash interfere.sh ~/rbq_data/enc/deep100m_1b
```

### 之后再定

- `deep1b` 和 `beir_msmarco`:取决于磁盘空间和 Preload 容量。
- GPU 基线:要在有 GPU 的机器上跑,第 0 步会显示 h1 有没有 GPU。

## 文件

| 文件 | 作用 |
|---|---|
| `vecio.py` | 读 fbin / fvecs / npy / hdf5 / parquet,大文件用 memmap,裁剪取前 N 条 |
| `encode.py` | 两遍流式编码(均值 → 旋转 + `sa_encode`),输出 SoA;`--qblocks-only` 换 k 重写参数块 |
| `gt.py` | 精确 top-100,分块计算;`--check` 与官方 GT 对照 |
| `hf_convert.py` | HuggingFace parquet → docs.npy / queries.npy(+ BEIR 的 id 与 qrels) |
| `fetch.sh` | 下载 + 整理;可续传 |
| `host_rabitq.c` | 主机 C 基线,与 kernel 同一算式;`--qbatch` 批处理 |
| `faiss_host.py` | FAISS `IndexRaBitQ` 基线,从 SoA 拼回码,不重新编码 |
| `eval.py` | recall@k;两份结果逐 id 对照 |
| `ir_eval.py` | 用 qrels 算 nDCG@10 |
| `rescore.py` | 卡返回 64 个候选,读原始向量重排 |
| `preload_probe.cpp` | 量 Preload Region 的槽位数和最大连续块 |
| `stream_triad.c`、`interfere.sh` | 干扰实验 |
| `h1_probe.sh` | 第 0 步:机器、卡、GPU、磁盘、占用情况 |
| `run_bench.sh` | 一个数据集的完整流程 |
