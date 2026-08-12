# 服务器操作手册 —— MX1P Hamming 扫描

开发机(dev machine)没有 XCENA 硬件,也没有 C++ 编译器(g++/clang/cmake/ninja 全无)。
本项目的每一行 kernel 代码和 host 驱动代码都是**盲写**的——从未在这台机器上编译过,
更没有在真实设备上跑过一次。本文档是这些代码第一次接触硬件时,执行者要照着走的
全部步骤。**请从头到尾按顺序执行,不要因为某一节看起来眼熟就跳到后面**——顺序本身
是经过设计的(见下方"执行顺序为什么不能打乱"),打乱顺序最典型的后果是把一次
16 倍性能误差当成了"基线"记录进论文。

## 0. 总览:已验证什么、没验证什么、三个必须提防的陷阱

### 0.1 已经在本地验证过的东西(不需要硬件,已完成,不要重新怀疑)

- **top-k 选择算法**(`model_kernel.py`,是 C++ kernel 的权威参考实现):
  - 200 次随机试验 + 定向用例 + 松阈值回归,全部通过。
  - 反证测试:把 `dist <= exactThresh` 这个过滤条件从代码里去掉,测试组确实会变红——
    证明测试真的在检验这件事,不是摆设。
  - 500 个强制候选溢出(overflow)配置,0 个错误答案。
- **`myRange` 分片规则**(kernel 内的任务切分逻辑)相对 `model_kernel.py::_task_range`
  交叉验证 217,669 次试验,0 处不一致。
- **C++ kernel 代码本身**(`mu_hamming.cpp`)被机械转录(transcribe)回 Python,
  与 `model_kernel.py` 逐项比对跑了 **306 次试验**(300 随机 + 6 个强制边界用例,
  包括 `numSigs=0`、`taskCount>numSigs`、`candCap=1` 溢出、`coarseThresh=0/256`),
  在 `ids`/`exactThresh`/`collected`/`overflow` 四项输出上 **0 处不一致**。
- **SWAR popcount** 实现对 20,137 个值验证,0 处失败。
- **A 组与 C 组用的是同一个距离定义**:`bench_host_baseline.py` 的 `bitwise_count`
  路径、它的 `np.unpackbits` 回退路径,与 `verify_host.py` 的 `hamming_all` 三者
  在 5000 个签名上逐位一致(0 不一致)。

**结论:如果在服务器上跑出错误的 top-k 结果,第一时间该怀疑的是环境、ABI、或
flush,而不是选择算法本身**——算法已经被差分测试(differential testing)到了很高
的置信度。

### 0.2 从未验证过的东西(只能在服务器上第一次见分晓)

任何实际编译、任何在 XCENA 卡上的执行、PXL host API 的真实调用形状——这些在开发机
上都无法验证,全部留到本手册。

### 0.3 执行顺序为什么不能打乱

1. **环境先行**(第 2 节 `env_check.sh`)——设备如果没在 devdax + Computing 模式,
   或 DKMS 驱动跟当前内核对不上,后面做什么都没有意义。
2. **编译**(第 3 节)。
3. **正确性必须先于性能**(第 5 节:模式 B 与 numpy 逐位比对 → 模式 C 的 top-k →
   强制溢出测试)——一个跑得快但算错的实现,看起来会像"性能胜出",但那是假胜出。
4. **拿到第一个正确结果之后,立刻扫 batchSize/numSub(第 6 节),不要拖到后面。**
   `活跃核数 ≤ ceil(taskCount/batchSize)`,而 `batchSize` 默认是 16,
   所以 `buildMap(fn, 2816)` 用默认参数只激活 `ceil(2816/16)=176` 个核,
   其余 2640 个核闲置——这是 **16 倍**的性能误差。**在完成这一步扫描之前测得的
   任何性能数字都必须作废**,不能进入 `RESULTS.md`。
5. **然后**才是 A/B/C 三组对比测量(第 7 节)与 popcount 双实现对比(第 8 节)。

### 0.4 三个必须提防的陷阱

1. **`flushHostCache` 诊断法则。** 症状三联征:结果错 **+** 每次运行结果不同
   **+** host 自己的自查(self-check)看起来正常 → 这三条同时出现,就是漏了
   `flushHostCache`,**不是**算法问题,**不是**编译器问题。这个模式坑过参考项目
   DRAN 大半天时间,第 5 节会再次强调。
2. **A 组基线必须在服务器上重新测量,且要和 B/C 组用同一台机器。**
   `RESULTS.md` 里现在的 A 组数字是在本地开发机上测的,与 B/C 组的设备(device)
   测量根本不是同一台机器——不同 CPU、不同内存带宽、不同 numpy build,
   跨机器数字不可比。任何跨机器算出的 A↔C 结论都是无效的,第 7 节会再次强调。
3. **PXL host API 是唯一在本地完全无法验证的接口。** SDK 只带 kernel 端的 MU
   头文件,`libpxl` 是服务器上单独装的包。所有 PXL 调用形状都是照抄已知能跑通的
   `dran_mu/dran_retrieve.cpp` 写的,但**如果编译失败,先怀疑 PXL 签名对不上,
   再怀疑 kernel 算法**——kernel 的算法已经被差分测试验证过(见 0.1),
   PXL 签名没有对应的本地验证手段。

---

## 1. 同步源码

```bash
./transfer.sh user@server:/home/you/work
```

只传源码,不传 `build/`、`*.mubin`、`*.bin`/`*.i32`/`*.ids`/`*.npy`、`__pycache__/`。
同步完成后脚本会提示下一步:

```
同步完成。在服务器上执行:
  cd /home/you/work/hamming_mu && bash env_check.sh
```

---

## 2. 环境校验(不过不要继续)

**Command:**
```bash
cd hamming_mu && bash env_check.sh
```

**Expected Output:**
- PCI 设备存在(`lspci -d 20a6:` 找到 XCENA 设备)
- `dkms status` 有匹配当前 `uname -r` 的 `mx_dma` 条目,且 `/dev/mx_dma/` 存在
- `daxctl list` 显示 `"mode":"devdax"`
- `xcena_cli device-info` 显示 `computable=Yes`
- `validate_host.sh` 无 FAIL

**若失败:**
- 若 `mode` 是 `system-ram`:
  ```bash
  sudo daxctl reconfigure-device --mode=devdax --force dax0.0
  ```
  (`--force` 必需:system-ram 模式下该内存已在线挂成 NUMA 节点,必须先下线。
  若 offline 失败,说明有进程已从该节点分配内存,用 `numactl -H` 查。)
- 若 `dkms status` 没有 `mx_dma` 条目:说明服务器换过内核,`mx_dma` 驱动需要针对
  当前内核重新构建(`env_check.sh` 自身在这一步会打印同样的提示)。
- 若 `computable=No`:固件不在 Computing 模式,不是驱动或 DAX 配置问题,需要
  厂商工具切换固件模式,不要在这一步继续排查软件栈。

**此步不通过,不要继续后面任何步骤。**

---

## 3. 编译

```bash
cd hamming_mu && chmod +x build.sh env_check.sh && ./build.sh
```

此时仓库里已经是最终状态:`mu_kernel/mu_hamming.cpp` 一次性导出三个 kernel 符号
(`hamming_scan_dists`、`hamming_scan_hist`、`hamming_merge_topk`),`hamming_scan.cpp`
一次性 `createFunction` 全部三个,所以**这是唯一一次编译**,不需要分阶段编译。

**Expected Output:**
- 编译无错误,无栈相关警告
- `ls -la hamming_mu/hamming_scan` 存在,且为本次新产物(时间戳更新)
- `ls -la hamming_mu/mu_kernel/mu_kernel.mubin` 存在,且时间戳更新(确认是本次
  重新生成,不是旧产物)
- 三个 kernel 符号都出现在产物里,可用第 4 节的 `llvm-objdump` 命令核实:
  ```bash
  $LLVM/bin/llvm-objdump -d hamming_mu/mu_kernel/mu_kernel.mubin \
    | grep -E '<hamming_scan_dists>|<hamming_scan_hist>|<hamming_merge_topk>'
  ```

**若失败,按下列条目逐一排查(不限于第一条命中就停,可能同时存在多个问题):**

- **`half_float 未声明` 之类错误** → 本项目为纯整数路径,`mu_hamming.cpp` 只应
  `#include "mu/mu.hpp"`,不应包含 `mu/vector.hpp` / `mu/vector/half.hpp` 之类。
  这是本项目最容易复现的一类错误,Task 1/3/6 的实现阶段都各自踩过一次。
- **参数不匹配 / 找不到符号** → 确认 `hamming_scan.cpp` 里三次 `createFunction`
  调用的名字,与 `mu_kernel/mu_hamming.cpp` 里三个 `MU_KERNEL_ADD` 完全一致:
  - `createFunction("hamming_scan_dists")` ↔ `MU_KERNEL_ADD(hamming_scan_dists)`,
    签名 `(const uint64_t* sigs, const uint64_t* query, uint64_t numSigs,
    int32_t* outDists)`,4 个参数;host 侧 `scanExec->execute(sigs, query,
    numSigs, dists)` 顺序、个数须与之一致。
  - `createFunction("hamming_scan_hist")` ↔ `MU_KERNEL_ADD(hamming_scan_hist)`,
    签名 `(sigs, query, numSigs, coarseThresh, candCap, outHist, outCand,
    outMeta)`,8 个参数。
  - `createFunction("hamming_merge_topk")` ↔ `MU_KERNEL_ADD(hamming_merge_topk)`,
    签名 `(hist, cand, meta, taskCount, candCap, topK, dists, outIds)`,8 个参数。
  - 以上均在 kernel 参数 ≤ 9 的限制内;若编译器报参数个数超限,先检查是否
    误传了额外参数,而不是删减已有参数。
- **PXL 相关类型 / 返回值不匹配**(`pxl::MemoryStatus`、`pxl::Result`、
  `job->buildMap`、`execute`/`synchronize` 等)→ 对照 `dran_mu/dran_retrieve.cpp`
  (已知可用的同类 host 驱动)核实用法,不要自行猜测 API——这是第 0.4 节陷阱 3
  提到的、全项目唯一没法本地验证的接口。
- **候选/直方图索引越界,或结果与 `hamming_mu/model_kernel.py` 的 `scan_hist` /
  `merge_topk` 不一致** → 以 `model_kernel.py` 为准逐行比对,尤其是这三处
  已经复现过的静默错误点:
  1. `cand[2j]=id, cand[2j+1]=dist` 的交错存储;
  2. `candCap*2` 的分片大小;
  3. merge 里 `dist <= exactThresh` 的过滤条件。
- **栈溢出相关警告** → `hamming_merge_topk` 唯一的大局部数组是
  `uint32_t total[BITS+1]`(257*4=1028 字节,远低于 64 KB/task 栈限制),先确认
  没有其他新增的局部数组吃掉了栈空间。
- **signed/unsigned 比较警告** → `ids[topK + 2] < (uint32_t)topK` 之类的比较里
  `topK` 已显式转 `uint32_t`,若警告指向别处,先看是不是新增代码里漏转了。
- **需要构建 SWAR 变体做对比**(第 8 节会用到)→ 加 `-DPOPCNT_SWAR`(临时改
  `mu_kernel/CMakeLists.txt` 的 `compile_options`,或在 `build.sh` 里传参)。
  **这一步现在先不用做**——先把默认(builtin)版本编译通过、跑过第 5 节的正确性
  关卡再说。

---

## 4. 反汇编:确认 popcount 是否映射到硬件指令

这一步的结论会在第 8 节(popcount 双实现对比)用到,提前在这里做,因为此时
`.mubin` 刚编译出来,环境变量也已经配好。

**Command:**
```bash
source /usr/local/mu_library/mu/script/min_llvm_version_env.sh
LLVM=/usr/local/mu_library/mu_llvm/$XCENA_LLVM_VERSION/$MU_REVISION
$LLVM/bin/llvm-objdump -d hamming_mu/mu_kernel/mu_kernel.mubin \
  | grep -A40 '<hamming_scan_dists>' | grep -iE 'cpop|call|jal' | head -20
```

**Expected Output / 如何解读:**
- 出现 `cpop` 指令 → `__builtin_popcountll` 被映射到了 RISC-V Zbb 扩展的硬件
  popcount 指令——默认(非 SWAR)变体应该更快或至少不慢于 SWAR。
- 出现 `call`/`jal` 跳到某个 `__popcount`/`__popcountdi2` 之类的库函数 →
  说明 builtin 退化成了软件实现(常见于目标 CPU 未声明 Zbb 扩展),此时手写的
  SWAR 变体(`-DPOPCNT_SWAR`)很可能更快,因为它是内联的位运算而非函数调用。
- **把观察到的是 `cpop` 还是 `call`/`jal`,以及具体指令文本,记下来**——第 8 节
  的 popcount 对比需要这个结论来解释两个实现的性能差异成因,不只是记两个数字。

---

## 5. 正确性验证(性能测量之前必须先过 —— 硬性顺序,不是建议)

一个跑得快但算错的实现,看起来会像"性能胜出",但那是假胜出。本节按 B → C →
强制溢出的顺序执行,任何一步不过,都不要往下走,更不要跳到第 6/7/8 节去测性能。

### 5.1 生成测试数据(若尚未生成)

```bash
python hamming_mu/verify_host.py gen -n 10000 -o /tmp/sig10k.bin
```

### 5.2 模式 B:与 numpy 逐位比对(本项目第一个真正的正确性关卡)

**Command:**
```bash
cd hamming_mu
./hamming_scan -n 10000 -t 64 --load /tmp/sig10k.bin --dump-dists /tmp/dev10k.i32
cd -
python hamming_mu/verify_host.py check --ref /tmp/sig10k.bin.dists.npy --dists /tmp/dev10k.i32
```

**Expected Output:**
```
PASS: 10000 个距离逐位一致
```

**若 FAIL,按此顺序诊断(不要跳过顺序,不要一上来就怀疑算法或编译器):**

1. **结果错 + 每次运行结果不同 + 主机自查(host 侧读回来看)正常** → 三条症状
   同时出现,就是 `flushHostCache` 问题,先查 flush 调用位置(写完数据后、
   preload/kernel 执行前;kernel 执行后、host 读结果前,各一次)。**不要**去
   怀疑算法或编译器——这正是第 0.4 节陷阱 1 的三联征。
2. 只错部分且稳定复现(每次运行结果一样但就是不对)→ 看 kernel 的 `hostPrintf`
   原始比特(hex)输出,与 host 侧同一个值的 hex 对比。浮点/十进制格式化会掩盖
   差异,原始比特不会。
3. 改了代码但行为没变(像是没生效)→ 加一行新的 `hostPrintf` 看它有没有出现,
   确认真的重新编译并重传了(不是跑到了旧的 `.mubin` 或旧的可执行文件)。

**待回填:** 实际 PASS/FAIL 输出、rep 计时、有效带宽数值。

### 5.3 模式 C:小规模验证 top-k 正确(端到端)

**Command:**
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 64 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/dev10k.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/dev10k.ids -k 100
```

（若 `/tmp/sig10k.bin` 和 `.query`/`.dists.npy` 还不存在,先用 5.1 的命令生成。）

**Expected Output:**
```
PASS: 100 个 id 距离全部 <= 第k名(...)
```

**如何解读 / 平局语义:** 判据是"返回的每个 id 的距离必须 <= 第 k 名的距离",
而**不是** id 集合与 numpy argsort 完全相同——计数排序与 numpy 在平局时选的
具体 id 可以不同,这不是错误。`check-topk` 同时检查数量、距离上界、id 唯一性
三项;三项皆过才算 PASS。

**若失败,按此顺序诊断:**
1. 数量不足(collected < k)但没有触发重试/警告 → 检查重试循环里 `ids[topK]` /
   `ids[topK+1]` / `ids[topK+2]` 三个状态槽的索引,是否与 `mu_hamming.cpp` 里
   `outIds[topK]=exactThresh`、`outIds[topK+1]=anyOverflow`、
   `outIds[topK+2]=collected` 完全对应——错位是最容易犯且最隐蔽的错误。
2. 结果每次运行不同 → 读 `ids` 之前是否每次都调用了
   `pxl::flushHostCache(ids, (topK+3)*sizeof(uint32_t))`;重试循环内、循环外、
   dump 之前都需要各自的 flush,漏一处就会读到设备写入前的陈旧数据(第 0.4 节
   陷阱 1 的另一种表现形式)。
3. 距离超过第 k 名 → 检查 `hamming_merge_topk` 里 `dist <= exactThresh` 的
   过滤条件是否被误删或改动(已复现过的静默错误点,见第 3 节);host 侧不应
   改动 kernel,但要确认 host 传给 kernel 的 `topK`/`candCap`/`taskCount`
   参数顺序没有传错,导致 kernel 侧算出错误的 `exactThresh`。

**待回填:** 实际 PASS/FAIL 输出、scan/merge 分别计时、是否触发过重试。

### 5.4 验证候选溢出路径真的会触发并自愈

**注:为什么是 `-t 2`,不是 `-t 4`。** 早先版本的本节用 `-t 4 -k 100`,声称
"候选数远超 candCap=topK*4=400,必然触发 overflow=1"——这个算术是错的。
`coarseThresh=118` 下,`P(Hamming ≤ 118) ≈ 0.1175`(随机 256 位签名,理论
二项分布);`-t 4` 时每 task 扫 2500 个签名,期望候选数 ≈ 2500×0.1175 ≈ 293,
**低于** 400 的 cap,不会溢出。用 `model_kernel.py` 独立复现(见下方"复现记录"):
`-t 4` 的每 task 候选数是 `[292,300,284,284]`,`overflow=0`,`collected=100/100`
——溢出/重试路径完全没有被执行到。这不是无伤大雅的文档笔误:操作者跑这一步、
看到 PASS、勾掉这一项,**溢出/重试路径就从未在硬件上跑过一次**——而这条路径
正是本项目对"成功门通过、结果却是错的"这一特征失败模式(见 0.1、第 3 节候选
溢出静默错误)的唯一硬件级防线。如果它没被触发过就上论文,这个防线形同虚设。

改用 `-t 2`:每 task 扫 5000 个签名,期望候选数 ≈ 5000×0.1175 ≈ 588,超过
400 的 cap,必然触发 `overflow=1` 且首轮收集不足,进而触发重试收敛。

**Command:**
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 2 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/ovf.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/ovf.ids -k 100
```

**Expected Output**(基于 `model_kernel.py` 独立复现,见下方"复现记录"——硬件
上的具体计时会不同,但 overflow/重试/收敛 这条状态机路径应一致):
- 首轮:每个 task 候选数打到 candCap=400 上限即截断,kernel 侧 `overflow=1`,
  首轮 `collected` 明显小于 100(复现中为 67/100)
- host 侧打印 `收集不足(67/100)-> coarseThresh 118 -> 精确阈值 109 重跑`
  (109 是本次种子下的实际第 100 名距离,不是固定值——不同种子/数据会不同,
  但"小于 118 且收敛后不再变化"这一形状应保持)
- 重跑一次后收敛:`collected=100/100`,不再触发下一轮重试
- 最终 `check-topk` **PASS**,返回的最大距离等于第 k 名距离

**关键判据是 `check-topk` PASS 且确实观察到 `overflow=1`**——本节的目的就是
确认这条路径被执行到,如果 `overflow` 从未变成 1,即便 `check-topk` PASS,
本节也没有测到它本该测的东西(见下方"为什么必须锁死这个配置")。

**复现记录(2026-08-12,`python hamming_mu/model_kernel.py` 之外,单独跑
`hamming_mu/model_kernel.py` 内的 `scan_hist`/`merge_topk`/`run_with_retry`
+ `verify_host.gen_sigs`/`hamming_all`,参数与本节命令一致:n=10000, k=100,
coarseThresh=118, candCap=topK*4=400,签名/query 与 `verify_host.py gen`
默认 seed 一致):**
```
n=10000 k=100 coarseThresh=118 candCap=400 true kth dist=109
P(d<=coarseThresh) empirical = 0.1160  期望候选数: t=4 时 290.0/task, t=2 时 580.0/task

=== taskCount=4 ===
per-task candCount: [292, 300, 284, 284]
per-task overflow  : [0, 0, 0, 0]
single-pass: overflow=0 collected=100/100 exactThresh=109
run_with_retry: attempts=0 final_collected=100/100
max returned distance=109  (kth=109)  OK=True

=== taskCount=2 ===
per-task candCount: [400, 400]
per-task overflow  : [1, 1]
single-pass: overflow=1 collected=67/100 exactThresh=109
run_with_retry: attempts=1 final_collected=100/100
max returned distance=109  (kth=109)  OK=True
```
`-t 4` 确认不溢出(与旧文档的断言相反);`-t 2` 确认溢出、收集不足 67/100、
一次重试收敛到 exactThresh=109、最终 100/100、返回的最大距离等于第 100 名
距离。这与审阅者报告的数字一致,独立复现通过。

**为什么必须锁死这个配置(防止本节再次"测了个寂寞"):** 本节的参数
(`n=10000 -k 100 -t 2`,配合代码写死的 `coarseThresh=118`、
`candCap=topK*4`)是**特意选到刚好会溢出**的——每 task 期望候选数
(≈588)明显超过 candCap(400)。这不是随手选的数字,是解出来的:
`n/t × P(d≤coarseThresh) > candCap` 才会稳定触发溢出。**如果以后改了
`n`、`k`、`t`、`coarseThresh` 的默认值,或者改了随机签名生成方式(改变
了距离分布),这个不等式可能不再成立,本节会像本次修复前那样安静地
"PASS 但没测到任何东西"。** 每次改动上述任一参数后,必须重新用
`model_kernel.py` 按上面的方法验证 `overflow` 确实变成了 1,而不是想当然
地认为旧参数还够用。

**若失败:**
- 若看到 `已在精确阈值 N 仍只收集到 x/100 -> candCap 太小`:说明重试已经收敛
  到精确阈值但 `candCap` 仍不够大,需要在 host 侧把 `candCap`(当前
  `topK*4`)调大后重跑,重试循环本身不会再有帮助——这是设计上的"重试无用"
  分支,不是 bug。
- 若 `check-topk` FAIL 但没有任何"收集不足"或"候选溢出"提示:先确认
  `-t 2` 确实生效(看 host 打印的 `tasks=2`),不是意外用了默认 taskCount。
- 若 5 次重试后仍未收敛(`for (int attempt = 0; attempt < 5; ...)` 用尽):
  记录 `coarseThresh` 的变化轨迹,检查是否在两个 `exactThresh` 值之间震荡而非
  收敛——按设计它应该单调收敛到某个不动点。

**待回填:** 是否观察到 overflow、host 侧走了哪条分支、`check-topk` 的最终
PASS/FAIL、以及若调大 candCap 重试过的话新的结果。

---

## 6. 扫描 batchSize / numSub(建立真实基线 —— 不可后移)

**CRITICAL:** 此步必须紧跟在第 5 节第一次拿到正确结果之后执行,在任何性能测量
之前。`setBatchSize` 默认为 16,导致 `buildMap(fn, 2816)` 仅激活
`ceil(2816/16) = 176` 个核,其余 2640 个核闲置 —— **16 倍性能差异**。若跳过或
延后此步,后续所有测量都建立在错误的基线上,必须作废重测。

### 6.1 扫 batchSize

**Command:**
```bash
cd hamming_mu && chmod +x sweep.sh && ./sweep.sh 10000000 2816
```

**Expected Output:**
- 6 行 batchSize 结果(b = 1, 2, 3, 4, 8, 16)
- 预期形状是**中间有谷**:b=1 因分派开销放大而慢,b=16 因并行度不足而慢,
  最优通常在 **b=3~4**
- 若最优就是 16,说明本负载的分派开销特征与 DRAN 参考项目不同 ——
  **记录下来,不要强行套用 DRAN 的结论**

**若失败:**
- 若全是 `FAIL`,检查 grep 模式是否与 `hamming_scan.cpp` 第 337 行的输出格式
  一致
- 若输出不是数字而是标点符号(`:`),说明 awk 字段索引错误 —— 应为 `$6`
  而非 `$5`(参考 `hamming_scan.cpp` 第 337 行的输出格式分析)

### 6.2 用最优 batchSize 扫 numSub

**Command:**
```bash
cd hamming_mu && BEST_B=<上一步的最优值> ./sweep.sh 10000000 2816
```

**Expected Output:**
- 4 行 numSub 结果(s = 4, 8, 16, 22)
- 通常在 s=8~16 附近有较平缓的区域

### 6.3 记录结果

**Command:**
```
编辑 hamming_mu/RESULTS.md,填入上述两个扫描的数字
```

**Expected Output:**
- batchSize 表格中的 6 个测量值和最优值
- numSub 表格中的 4 个测量值和最优值
- 可选:计算有效带宽 = (scannedGB) / (min_ms / 1000) GB/s,其中 scannedGB 在
  `hamming_scan.cpp` 输出中显示

**待回填:** batchSize 和 numSub 的实测值与最优值选择。

---

## 7. A/B/C 三组对比测量

### 7.1 A 组 host 基线(必须在服务器上重新测量)

**背景:** 本地开发机(dev machine,Windows,Intel i9-9880H,numpy 2.4.6)已用
`hamming_mu/bench_host_baseline.py` 跑通 A 组基线(N=1M 与 N=10M),确认了实现
正确(与 `verify_host.py` 的 `check-topk` 交叉验证 PASS)、`bitwise_count` 路径
可用、mean Hamming 距离约 128(符合随机签名预期)。但这些数字**只是 provisional
的健全性检查**,不能进入论文的 A vs B/C 对比 —— 本项目的测量纪律要求 A 组基线
必须与 B/C 两组的设备(device)测量在**同一台机器**上跑,跨机器数字不可比
(不同 CPU、不同内存带宽、不同 numpy build)。这正是第 0.4 节陷阱 2。

**Command:**
```bash
cd hamming_mu
python bench_host_baseline.py -n 1000000 -k 500 --reps 5
python bench_host_baseline.py -n 10000000 -k 500 --reps 5
python bench_host_baseline.py -n 100000000 -k 500 --reps 5
```

**Expected Output:**
- 三个 N 各自的最小耗时(ms)与有效带宽(GB/s)
- 确认输出中 `bitwise_count=True`(即用了原生 popcount 路径,而非
  `np.unpackbits` 查表法回退;若服务器 numpy 版本 < 2.0 导致回退,需在
  `RESULTS.md` 中明确记录用了哪条路径,两条路径性能可能有数量级差异)

**若失败:**
- N=100M 若 OOM 或换页明显(比 N=10M 单条数据的线性外推慢很多):降到能安全跑的
  最大 N,并在 `RESULTS.md` 中注明本机的规模上限(dev 机上因可用内存仅约 3.2 GB
  已跳过 N=100M,原因需一并记录,不要沉默跳过)。

**产出去向:** 用服务器上测得的 A 组数字**替换** `RESULTS.md` 中标注为
"dev 机 provisional"的三个单元格(1M/10M/100M 的 A host-only 行),并去掉
"provisional / 非最终对比口径"的标注;之后"三组对比"表格与"诚实边界"章节里
关于 A↔C 机器不一致的警告段落才能删除或改写为"已同机测量,数字可比"。

### 7.2 B、C 两组(用第 6 节的最优参数)

**Command:**
```bash
cd hamming_mu
B=<第6.1节最优batchSize>; S=<第6.2节最优numSub>
for N in 1000000 10000000 100000000; do
  echo "=== N=$N mode B ==="
  ./hamming_scan -n $N -k 500 -t 2816 -b $B -s $S --mode B --reps 5 | tail -20
  echo "=== N=$N mode C ==="
  ./hamming_scan -n $N -k 500 -t 2816 -b $B -s $S --mode C --reps 5 | tail -20
done
```

**Expected Output:**
- 6 组输出(N × {B,C})。N=100M 时 arena 约 3.2 GB(mode B 另需 400 MB 距离数组),
  需确认 Preload Region 有足够 1 GB 槽位;若 `preloadMemory failed`,降到 N=50M
  并在 `RESULTS.md` 中注明规模上限。

**产出去向:** 填入 `RESULTS.md` "三组对比"表中对应 N × {B,C} 的"每查询延迟"
单元格。

---

## 8. popcount 双实现对比(builtin vs SWAR)

**Command:**
```bash
cd hamming_mu
# 默认 builtin 版已在第 7.2 节测过,现在编 SWAR 版
sed -i 's/^set(compile_options/set(compile_options\n    -DPOPCNT_SWAR/' mu_kernel/CMakeLists.txt
./build.sh
./hamming_scan -n 10000000 -k 500 -t 2816 -b $B -s $S --mode C --reps 5 | grep -E 'scan \(min|有效带宽'
# 验证 SWAR 版同样正确 —— 不能只看它"更快",必须先过 check-topk
./hamming_scan -n 10000 -k 100 -t 64 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/swar.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/swar.ids -k 100
```

**Expected Output:**
- SWAR 版 `check-topk` 必须同样 `PASS`。**这是本步骤的门槛,不是可选项**:
  DRAN 项目的教训是手写整数路径在 `-O3` 下不能假定必然正确 —— 一个跑得快但算错
  的 SWAR 实现,如果不先过正确性关卡,看起来会像"性能胜出",但那是假胜出。
  只有 `check-topk` PASS 之后,SWAR 版的计时数字才可以写进 `RESULTS.md` 的
  popcount 对比表;若 FAIL,先修 kernel,不要记录它的计时。
- 记录两版(builtin / SWAR)的 scan 耗时与有效带宽差异。
- 把第 4 节反汇编步骤得到的"是 `cpop` 硬件指令还是 `call`/`jal` 库函数调用"
  结论填进 `RESULTS.md` 该表下方的"反汇编观察"一行 —— 这决定了两个实现差异
  的成因解释,不只是记两个数字。

**测完后恢复 builtin 版(若 builtin 更快,或作为默认发布配置):**
```bash
cd hamming_mu && sed -i '/-DPOPCNT_SWAR/d' mu_kernel/CMakeLists.txt && ./build.sh && cd -
```
恢复后建议重跑一次第 5.2 节的 `check --ref ... --dists ...` 逐位比对,确认
`sed -i` 撤销没有把 `CMakeLists.txt` 改坏(比如误删了原有别的编译选项)。

**产出去向:** 填入 `RESULTS.md` "popcount 实现对比"表(builtin 与 SWAR 两行的
`scan (ms)`、`有效带宽 (GB/s)`、`正确性` 三列)与其下方的反汇编观察一行。

---

## 9. 把结果写回 RESULTS.md,传回开发机

确认 `RESULTS.md` 中所有 `<填写>` 占位符都已被第 5~8 节的实测数字替换,
A 组的"dev 机 provisional"标注按第 7.1 节说明已替换或改写,"诚实边界"章节的
机器一致性警告段落已按实际情况更新或删除,然后把整个 `hamming_mu/` 目录传回
开发机(或直接把改动过的 `RESULTS.md` 拷回),供后续整理与提交。
