# rabitq_mu —— RaBitQ 两阶段扫描下沉到 XCENA MX1P

RAGDWM 的二值预筛(binary pre-filter)从 SimHash 换成 RaBitQ 后的设备端实现。
动机与前置实验见 RAGDWM 的 `report_rabitq_vs_simhash.py`(同字节数下所需候选少
9–35 倍)和 `report_rabitq_refine.py`(大库上第二阶段读流量只占第一阶段的 1–2%)。
放在 `engramme/` 下与 `hamming_mu/`、`dran_mu/` 平行,只是为了复用 XCENA SDK 与构建方式。

## 计算是怎么分的(对齐 FAISS 1.15.1 源码)

| | 第一阶段(全部 N 条) | 第二阶段(只对通过误差界的) |
|---|---|---|
| 读 | 符号码 48 B + 因子 12 B(d=384) | 附加码 + 因子 8 B(4 bit 时 152 B) |
| 算 | 每 64 位字:1 次 popcount + qb 次(按位与 + popcount) | d 维:浮点查询 × 整数码,乘加 |
| 访存 | 顺序扫描 | 随机读 |

精算判定:`下界 = 粗估距离 − f_error × g_error`,下界 < 当前第 k 名才精算。

## 设备上的关键设计:全局阈值 τ(G 模式)

FAISS 的平铺搜索是单线程顺序扫描,阈值随堆逐步收紧。直接并行化(每个 task 各自的堆,
T 模式)在 2816 个 task 下会退化:每个 task 分到的向量太少,堆填不满,几乎全部被精算。
G 模式先求全局 τ = 所有向量上界 `粗估 + f_error × g_error` 中第 k 小的值,再只精算下界 < τ 的:

| FiQA(5.7 万条),4 bit | 精算比例 | 第二阶段字节 / 第一阶段 | 召回@10 |
|---|---|---|---|
| 1 task(FAISS 顺序扫描) | 0.76% | 1.9% | 0.950 |
| 2816 task 各自的堆(T) | 90% | 229% | 0.950 |
| **2816 task 全局 τ(G)** | **0.68%** | **1.7%** | **0.950** |

G 分三步:`rabitq_bounds`(每条写 4 B 下界,每 task 记 k 个最小上界)→ `rabitq_tau`
(单 task 求 τ)→ `rabitq_refine`(只读 4 B 下界,不重读 60 B 符号码),最后 `rabitq_merge`。

## 文件

| 文件 | 作用 |
|---|---|
| `rabitq_ref.py` | 参考模型,逐行对齐 FAISS;kernel 照它写 |
| `test_ref_vs_faiss.py` | 参考模型 vs FAISS(开发机可跑) |
| `test_transcription.py` | 把 kernel 逐句转写成 Python,与参考模型比(开发机可跑) |
| `mu_kernel/mu_rabitq.cpp` | 8 个 kernel:诊断 `ints / est1 / full`,T 模式 `topk`,G 模式 `bounds / tau / refine`,共用 `merge` |
| `rabitq_scan.cpp` | host 驱动,`--mode I / E / F / T / G` |
| `prep_rabitq.py` | FAISS 编码 → SoA 二进制文件;`--random-codes` 只为测带宽 |
| `verify_rabitq.py` | 校验设备输出;`--simulate` 在开发机上模拟设备输出,测整条链路 |
| `run_rabitq.sh` | h1 上一键:正确性(I→E→F→T→G 分层)+ 第一阶段带宽 + batchSize 粗扫 |
| `vecdata.py` | 向量加载与子集抽样(prep 与规模验证共用,保证同一批文档) |
| `scale/encode_msmarco.py` | bge-small 编码 BEIR MS MARCO(884 万),可续跑 |
| `scale/scale_kprime.py` | 规模验证:所需候选数(SimHash vs RaBitQ)与三种两阶段方式的精算数 |
| `scale/run_scale.sh` | h1 上一键:下载 → 编码 → 10 万 / 100 万 / 全部;`DEVICE=1` 另在卡上跑 G/T |

## 已在开发机上验证(2026-10-09)

- 参考模型 vs FAISS(SciFact 50 查询、FiQA 20 查询,nb = 1/2/4):第一阶段相对误差 ≤ 2e-7,
  第二阶段 ≤ 4e-7,两阶段 top-10 与 FAISS 一致(FiQA 4 bit 2816 task 时 1/20 不同 ——
  查明是 FAISS 全局阈值收紧后误差界剪掉了一个更近的候选,分段扫描反而找到了它)
- kernel 转写 vs 参考模型:整数 0 不一致,第一阶段逐位相同,第二阶段 ≤ 7e-7,
  每 task 的 top-k、精算次数、归并结果全部一致;G 模式三个 kernel 的 τ 逐位相同
- 整条文件链路(prep → 模拟设备输出 → verify)通过;篡改(改整数、改距离、写 NaN、换 id、
  用 T 的输出冒充 G)全部被 verify 抓出 —— 后两种是在本地测试中发现 verify 有漏洞后补上的
- scale_kprime.py 在 FiQA 上复现了 RAGDWM 原实验的数字(所需候选、精算数)

## 没有验证、只能上卡才知道的

- **编译**:开发机没有 C++ 编译器,kernel 与 host 都是盲写的
- **标量浮点**:DRAN 记录过手写标量浮点循环在本工具链 `-O3` 下算错。E/F 模式就是为此设的;
  若 I 对而 E/F 错,优先怀疑这一点
- **性能**:第一阶段每字的计算量约是 Hamming 的 4 倍,最优 batchSize 未必还是 16

## h1 上怎么跑

```bash
cd ~/engramme && git pull origin master
cd rabitq_mu
pip install numpy faiss-cpu numba
mkdir -p emb && scp <开发机>:~/.cache/ragdwm/emb/*.npz emb/   # 可选:真实语料的正确性检查
bash run_rabitq.sh
```

跑完把 `results_*/summary.txt` 发回来。

规模验证(另一个目录,可以先跑,不依赖卡):

```bash
cd ~/engramme/rabitq_mu/scale
pip install torch sentence-transformers      # 若还没有
bash run_scale.sh                             # 下载 → 编码(A100 约半小时)→ 统计
DEVICE=1 bash run_scale.sh                    # run_rabitq.sh 通过之后,再在卡上跑 MS MARCO
```
