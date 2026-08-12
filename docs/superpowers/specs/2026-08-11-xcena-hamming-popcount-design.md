# MX1P 近内存 Hamming 扫描(XOR + POPCOUNT)· 设计文档

2026-08-11 · 目标:为 DWM/RAGDWM 论文提供"近内存 Hamming 粗筛 vs host"的硬件对比数据点。

依据 `xcena-mx1p-playbook.md`(DRAN 项目 2026-07~08 沉淀)。本设计复用其
"分条扫描 + 每 task 局部结果 + 单 task 归并"的形状,不复用其迭代反馈闭环。

---

## 1 · 目标与非目标

**目标**:拿到可信、可复现的三组对比数字,写进论文的硬件章节。
重心是**测量严谨**,不是 kernel 极致优化。

**非目标(明确 YAGNI)**:
- 不做迭代反馈闭环(DRAN 需要,本负载单轮即可)
- 不做 bit-sliced 布局(Hamming 需全部 256 位,无法提前剪枝)
- 不做多查询批处理(先把单查询数字做实)
- 不追 268 GB/s 带宽上限(目标是可信对比)

---

## 2 · 三组对比

同一份签名数据、同一查询集。三组都是**精确 Hamming**,无近似,
因此 **recall 天然恒等** —— 性能对比中质量是常量,排除"降质换速度"的质疑。

| 组 | 做法 | 链路流量/查询 |
|---|---|---|
| **A** host-only | numba 融合 popcount,数据在主机 | 整库需入主机 |
| **B** 设备扫描 + host 选 | 设备算距离,回传 N 个距离,host 选 top-k | query 32B + N×2B |
| **C** 设备扫描 + 设备选 | 设备算距离 + 计数排序 top-k,只回 k 个 id | query 32B + k×4B ≈ 2 KB |

- **A ↔ C** = 近内存的价值
- **B ↔ C** = selection 下沉的价值

B 组只需在 C 的 kernel 上加一个"跳过 top-k、直接写 dist 数组"的编译开关,
近乎零额外成本,却买到一个独立论据。

**A 组必须在同一台机器重测**,不得引用 RAGDWM 旧的 2.11 ms@200k(跨机器不可比)。

---

## 3 · 数据与布局

**数据**:合成随机 256 位签名,规模 1M → 10M → 100M。
(100M × 32B = 3.2 GB,才吃得满卡;Hamming 每条目仅 32B,比 DRAN 的
fp16 列 2 KB 省 64×,所以需要更多条目才能吃满。)

**签名布局**:每签名 4×uint64 连续,签名 i 占 `[i*32, i*32+32)` 字节。
每 task 扫一段连续签名 → 顺序读,匹配 DRAM 最擅长的模式(playbook §5.5)。

**单 arena**(playbook §2.1,preload 粒度为 1 GB 对齐 + 1 GB 整数倍,
每 buffer 单独 preload 会让 4 KB 输出吃掉整个 GB 配额):

```
[ 签名库 N×32B ][ query 32B ][ 直方图 T×257×4B ]
[ 候选 T×C×8B ][ candCount/overflow T×8B ][ 输出 k×4B ]
```

N=100M 时 arena 约 4 GB,**整体 preload 一次**。

---

## 4 · 两个 kernel

### kernel 1 `scan_hamming` — taskCount = 分条数

每 task 扫一段连续签名。每个签名做 4 次 XOR + popcount 累加 → 距离 ∈ [0, 256]。然后:

1. 累加到该 task 私有的 **257 桶直方图**(距离是有界整数 → O(N) 计数排序的基础)
2. 将 `dist ≤ 粗阈值` 的候选**连同它的距离**写入该 task 的候选分片:
   交错存放 `cand[2j] = id`、`cand[2j+1] = dist`
3. 维护该 task 的 `candCount` 与 `overflow` 标志

每 task 只写自己的分片 → **天然无竞争**(playbook §4:Map 内无跨 task 同步)。

B 组变体:跳过 1/2/3,直接把 dist 写进全局 dist 数组。

### kernel 2 `merge_topk` — taskCount = 1

1. 归并所有 task 的 257 桶直方图
2. 从距离 0 往上累加,定位第 k 名所在桶 → **精确距离阈值 `exactThresh`**
3. 扫候选分片,**只收 `dist ≤ exactThresh` 的候选**,取满 k 个写入输出

**为什么必须两个 kernel**:归并需看到所有 task 的直方图,而 Map 内无同步。
playbook §4 明确要求"每 task 写分片 + 再启 taskCount=1 的 kernel 归并"。

### ★ 为什么候选必须带上距离(一个已复现的静默错误)

**初版设计里候选只存 id,merge 按索引顺序取前 k 个 —— 这是错的,而且是静默的。**
候选只满足 `d ≤ coarseThresh`;当 `coarseThresh > exactThresh` 时,
按索引序取前 k 个会取到粗阈值宽带里的任意 id,而不是最近的那些。

实测复现(n=2000, k=50, taskCount=4, coarse=148, cap 足够大):

```
overflow=0  collected=50  ->  成功门通过
exactThresh=113 = 真实第k名距离
但返回 id 的最大距离 = 145   ✗ 错误
```

**成功门通过、结果却是错的** —— 正是本项目最警惕的静默失败。
根因:候选缓冲不存距离,merge 物理上无法按 `exactThresh` 过滤。

**修法(Option B):候选带上距离,merge 真的按 `exactThresh` 过滤。**
由此得到一条关键性质:

> 过滤后每个返回 id 都满足 `d ≤ exactThresh = 第k名距离`,
> 所以**溢出不再可能产生错误结果**,只可能导致收集不足。

失败模式从"静默返回错的"降级为"响亮地收集不够",后者可检测、可自愈。

### 候选溢出与阈值自愈

直方图统计**全部**签名,不受候选缓冲容量影响,所以 `exactThresh` 始终可算。
主机侧的重试规则因此非常简单:

| 观察 | 含义 | 动作 |
|---|---|---|
| `collected ≥ topK` | 结果有效(全部 `d ≤ exactThresh`) | 完成,**即便 overflow=1** |
| `collected < topK` | 粗阈值过紧,或溢出丢掉了合格项 | 令 `coarseThresh = exactThresh` 重跑 |
| 在 `coarseThresh == exactThresh` 仍不足 | `candCap` 太小 | 增大 `candCap` 重跑 |

两个方向都收敛到 `exactThresh`:过紧则放宽,过松则收紧。
`overflow` 标志退化为**诊断信息**而非错误条件。

---

## 5 · popcount 双实现

MU 核只有 fp16/fp32 向量引擎,无专用 popcount 向量指令。两种实现实测对比:

- **`POPCNT_BUILTIN`**:`__builtin_popcountll` —— RISC-V 有 Zbb 扩展时映射到 `cpop`
- **`POPCNT_SWAR`**:经典 5 步掩码分治法,不依赖硬件指令

用 `constexpr` 开关编出两个 `.mubin`(kernel 参数上限 9 个,不浪费在开关上,
playbook §5.2)。**先看反汇编确认 builtin 是否真用上硬件指令** ——
若退化为软件实现,SWAR 可能更快。

注:playbook §5.6 记录了"手写标量循环替代 dotProduct 无收益",而
`mu_dran.cpp` 源码注释记录得更严重:**手写标量循环在该工具链 -O3 下产生了错误的
点积**(host 0.001360 vs device -0.245,同一份字节),速度还一样。

这条对本项目的含义:那是**浮点**路径的结论,popcount 是纯整数位运算,
向量引擎本就不支持,必须手写,无法沿用"用向量引擎就行"的结论。
但它提高了风险等级 —— **两种 popcount 实现都必须在小规模上对 numpy 逐位验证
通过后才能用于性能测量**,不能假定整数路径在 -O3 下必然正确。

---

## 6 · 主机侧驱动与测量

骨架照搬 `dran_mu/dran_retrieve.cpp`(单 arena、flushHostCache、命令行旋钮),
**测量部分重写**。

### flushHostCache(playbook §3,最大的坑)

- 主机写完、kernel 启动**之前**必须 flush
- kernel 跑完、主机读结果**之前**必须 flush(失效方向)
- 从第一版就写进骨架,视为**正确性的一部分**,不是优化

### 三段时间分开报告

| 段 | 内容 | 计入稳态 |
|---|---|---|
| 一次性初始化 | 填数据 + flushHostCache + preloadMemory | **否**(数据常驻,稳态不重复) |
| 每查询延迟 | 两个 kernel 的 execute + synchronize | **是** |
| 结果回传 | flush + 读 k×4B | **是** |

**报告口径:稳态每查询延迟(数据已在卡上),初始化成本单独列一行明写。**
诚实且对我方有利,但必须明写不得隐藏。

### 核心指标

```
链路流量/查询   C: 32B + k×4B ≈ 2 KB
                B: 32B + N×2B = 200 MB (N=100M)
                A: 整库 3.2 GB 需入主机
设备内实扫      N×32B = 3.2 GB
有效带宽        3.2 GB / 扫描耗时  → 对照 268 GB/s
```

playbook §7 的教训:**链路流量比总时间更有说服力**(DRAN 是 14.8 KB vs 6.15 GB,
四个数量级),做对比论证时优先摆这个数。

同时应诚实引用 §7.3:近内存**不赢在算得快**(A100 4.2 ms vs MX1P 203.9 ms,
差 48×),赢在数据不用搬 —— 论证要建立在容量与并发上。

### 测量纪律

- 每配置跑 ≥5 次取最小值(与 `report_latency.py` 口径一致)
- 计时前先 warmup 一轮
- **跑通后立刻扫 batchSize**(§10.2):`b ∈ {1,2,3,4,8,16}`,再扫 `numSub`
  (默认 batchSize=16 → 仅激活 `ceil(taskCount/16)` 个核,16 倍差距会污染所有基线)

---

## 7 · 正确性验证

性能测量**之前**必须通过:小规模(N=10k)设备结果与 numpy
`count_nonzero(sig != q, axis=1)` **逐位一致**。`verify_host.py` 负责生成
合成签名、算 numpy 参考答案、比对设备输出。

**内建调试路径**(playbook §10.4):每个 kernel 从第一版就带
`hostPrintf` 打**原始比特**的路径,用 `taskIdx==0` 守卫。
浮点/十进制格式化会掩盖差异,原始比特不会。

**故障判据**(§3):结果错 + **每次运行结果不同** + 主机自查正常
→ 三条同时出现,先查 `flushHostCache`,不要去怀疑算法或编译器。

---

## 8 · 文件组织

新建 `engramme/hamming_mu/`,与 `dran_mu/` 平行,**不改动 DRAN 代码**:

```
hamming_mu/
  mu_kernel/mu_hamming.cpp    scan_hamming + merge_topk
  hamming_scan.cpp            主机驱动(单 arena / flush / 旋钮 / 计时)
  CMakeLists.txt              照抄 dran_mu 改名
  build.sh
  verify_host.py              合成签名 + numpy 参考 + 比对
```

---

## 9 · 交付顺序

每步都是可验证检查点,不跳步:

1. **环境** — `validate_host.sh` 通过 + `computable=Yes` + devdax
   (§1.1:devdax 不跨重启保留,重启后需
   `daxctl reconfigure-device --mode=devdax --force`)
2. **骨架** — kernel 只算距离写 dist 数组(即 **B 组**),N=10k,对 numpy 逐位一致
3. **扫 batchSize** — 定下真实基线(必须在此步,不能推后)
4. **加 top-k** — 计数排序 + 溢出处理(**C 组**),仍在小规模验对
5. **上规模** — 1M → 10M → 100M,测三组
6. **popcount 对比** — 双实现实测 + 出表

顺序理由:B 组是 C 组的子集,先做 B 既拿到对照组数据又是最简可验证起点;
第 3 步插在中间是 playbook 的血泪教训,晚扫会让后续测量全部建在错基线上。

---

## 10 · 风险与对策

| 风险 | 对策 |
|---|---|
| builtin popcount 退化为软件实现 | 已计划双实现实测,看反汇编确认 |
| 候选缓冲溢出丢结果 | 直方图算精确阈值 + 回退重扫(§4) |
| 32B/条目太小,访存效率低于 DRAN 的 2KB/列 | 需实测有效带宽;若偏低,考虑每 task 多签名批量读 |
| 只用上 1/16 核 | 第 3 步强制扫 batchSize |
| 结果随机错 | flushHostCache 判据(§7) |
