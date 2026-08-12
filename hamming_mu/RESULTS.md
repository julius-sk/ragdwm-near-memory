# MX1P Hamming 扫描 —— 测量记录

## 环境

- 日期:<填写>
- 机器:<填写,与 A 组基线必须同一台>
- SDK:v1.4.9
- popcount 实现:builtin / SWAR(见 Task 3 Step 3 的反汇编观察)
- 反汇编观察:<cpop 指令 / 库函数调用>

## batchSize 扫描 (N=10M, tasks=2816)

| batchSize | scan (ms) | 有效带宽 (GB/s) |
|---|---|---|
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 8 | | |
| 16 (默认) | | |

**最优 batchSize = <填写>**

## numSub 扫描 (batchSize = 最优)

| numSub | scan (ms) |
|---|---|
| 4 | |
| 8 | |
| 16 | |
| 22 | |

**最优 numSub = <填写>**

## 后续所有测量使用的参数

`-t 2816 -b <最优> -s <最优>`

## A 组 host 基线(dev 机器,provisional 口径 —— 尚未在服务器复测)

**重要:以下数字是在本地开发机(dev machine)上用 `hamming_mu/bench_host_baseline.py`
(纯 numpy)测得的,机器与 Task 3~7 的设备(device)测量**不是同一台**。
按本项目的测量纪律(measurement discipline),A 组基线必须与 B/C 组的设备测量在
同一台机器上跑才可比较——跨机器数字不可比。下面的数字仅作为实现正确性与数量级
的**健全性检查(sanity check)**,不得直接放入论文的 A vs B/C 对比结论。
**A 组必须在服务器上重跑**,见 `RUNBOOK.md` 的"待服务器验证"章节。

**测量环境(dev 机器,非最终发布环境):**
- 日期:2026-08-12
- 机器:开发机(Windows 10 Pro),Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz(8 核 16 线程)
- Python:3.12.10
- numpy:2.4.6
- popcount 路径:`np.bitwise_count`(numpy >= 2.0 原生支持,本机可用,已走该分支,
  未退回 `np.unpackbits` 查表法)

**实测结果(reps=5,取最小值;k=500):**

| N | 最小耗时 (ms) | 有效带宽 (GB/s) | 需入主机的数据 |
|---|---|---|---|
| 1,000,000 | 42.296 | 0.8 | 0.032 GB |
| 10,000,000 | 400.561 | 0.8 | 0.320 GB |
| 100,000,000 | 未测(dev 机可用内存仅约 3.2 GB,不足以安全容纳 100M×32B 签名数组
  +计算所需的临时数组,存在 OOM/换页风险,故未执行,避免产生不可靠数字) | — | — |

**终端原始输出**(N=10,000,000,`python hamming_mu/bench_host_baseline.py -n 10000000 -k 500 --reps 5`):
```
  rep 0: 430.979 ms
  rep 1: 411.557 ms
  rep 2: 400.561 ms
  rep 3: 414.013 ms
  rep 4: 409.453 ms

A 组 host 基线  N=10000000  k=500
  最小耗时      : 400.561 ms
  有效带宽      : 0.8 GB/s
  需入主机的数据: 0.320 GB
  numpy         : 2.4.6  bitwise_count=True
```

**正确性交叉验证(cross-check,against 已 review 的参考实现 `verify_host.py`):**
用 `verify_host.py gen -n 5000 --seed 7` 生成的签名与 query,分别用
`bench_host_baseline.scan_topk()` 与官方 CLI `verify_host.py check-topk` 校验,结果:
```
PASS: 100 个 id 距离全部 <= 第k名(112);与参考重合 98/100
```
即基线选出的 top-100 id 全部满足 `距离 <= 第100名距离`(与 `check-topk` 判据一致),
与参考实现重合 98/100——2 个不重合是精确并列(tie)在 argpartition 与 stable argsort
之间的边界选择差异,不是正确性问题(与 Task 7 RUNBOOK 记录的平局语义一致)。
另外用 200,000 个随机 256 位签名验证了全局分布:平均 Hamming 距离 127.99(理论期望 128),
min 94 / max 163,量级符合随机签名的预期。

## 三组对比(k=500,参数 -t 2816 -b <最优> -s <最优>)

**以下表中,A 组的 1M/10M 单元格来自上面 dev 机器 provisional 测量(已标注),
100M 及全部 B/C 组数据需设备(device)与服务器执行,当前留空,严禁编造。**

| N | 方法 | 每查询延迟 (ms) | 链路流量/查询 | 设备内实扫 |
|---|---|---|---|---|
| 1M | A host-only | 42.296 (dev 机 provisional,非最终对比口径) | 32 MB 需入主机 | — |
| 1M | B 设备扫+host选 | <填写> | 4.0 MB | 32 MB |
| 1M | C 全下沉 | <填写> | 2.0 KB | 32 MB |
| 10M | A host-only | 400.561 (dev 机 provisional,非最终对比口径) | 320 MB 需入主机 | — |
| 10M | B 设备扫+host选 | <填写> | 40 MB | 320 MB |
| 10M | C 全下沉 | <填写> | 2.0 KB | 320 MB |
| 100M | A host-only | <填写>(dev 机未测,内存不足,需服务器补测) | 3.2 GB 需入主机 | — |
| 100M | B 设备扫+host选 | <填写> | 400 MB | 3.2 GB |
| 100M | C 全下沉 | <填写> | 2.0 KB | 3.2 GB |

**读法:**
- **A ↔ C** = 近内存的价值
- **B ↔ C** = selection 下沉的价值

## popcount 实现对比 (N=10M, mode C)

| 实现 | scan (ms) | 有效带宽 (GB/s) | 正确性 |
|---|---|---|---|
| `__builtin_popcountll` | <填写> | <填写> | <填写> |
| SWAR 5 步掩码 | <填写> | <填写> | <填写> |

反汇编观察:<cpop 硬件指令 / 库函数调用>

## 诚实边界

- 延迟为**稳态**口径:数据已在卡上,不含一次性 fill/flush/preload
  (N=100M 时 flush 约 <填写> ms、preload 约 <填写> ms,均为一次性)。
- 合成随机签名。Hamming 距离与内容无关,故合成数据对**延迟**测量无偏;
  但 recall 相关结论不在本实验范围。
- 三组均为**精确 Hamming**,recall 恒等,性能对比中质量是常量。
- 近内存**不赢在算得快**:同类扫描 GPU 单次更快,优势在数据不搬、
  链路流量低几个数量级,以及由此带来的并发扩展性。
- 有效带宽 vs 268 GB/s 上限:<填写>%。若在 10~30% 区间,说明 kernel 仍有优化空间。
- **A 组数字的机器一致性(machine consistency)警告:** 本文档中标注为
  "dev 机 provisional"的 A 组数字与本文档其余(B/C 组、popcount 对比)在
  **服务器**上测得的数字**不在同一台机器上**,不可直接比较、不可用于计算
  A↔C 或任何比值结论。在服务器补测 A 组之前,"A ↔ C = 近内存的价值"这一
  读法暂不成立,仅供工程 sanity check 参考。
