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
