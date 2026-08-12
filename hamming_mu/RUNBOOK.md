# RUNBOOK

## 待服务器验证

### Task 8 Step 2: A 组 host 基线必须在服务器上重跑

**背景:** Task 8 已在本地开发机(dev machine,Windows,Intel i9-9880H,numpy 2.4.6)
用 `hamming_mu/bench_host_baseline.py` 跑通 A 组基线(N=1M 与 N=10M),确认了实现
正确(与 `verify_host.py` 的 `check-topk` 交叉验证 PASS)、`bitwise_count` 路径可用、
mean Hamming 距离约 128(符合随机签名预期)。但这些数字**只是 provisional 的
健全性检查**,不能进入论文的 A vs B/C 对比——本项目的测量纪律要求 A 组基线必须
与 B/C 两组的设备(device)测量在**同一台机器**上跑,跨机器数字不可比(不同 CPU、
不同内存带宽、不同 numpy build)。

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

---

### Task 8 Step 3: 跑 B、C 两组(用 Task 5 的最优参数),填三组对比表的 B/C 单元格

**Command:**
```bash
cd hamming_mu
B=<Task5最优batchSize>; S=<Task5最优numSub>
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

**产出去向:** 填入 `RESULTS.md` "三组对比"表中对应 N × {B,C} 的"每查询延迟"单元格。

---

### Task 8 Step 4: popcount 双实现(builtin vs SWAR)对比,填 popcount 对比表

**Command:**
```bash
cd hamming_mu
# 默认 builtin 版已在 Step 3 测过,现在编 SWAR 版
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
  DRAN 项目的教训是手写整数路径在 `-O3` 下不能假定必然正确——一个跑得快但算错的
  SWAR 实现,如果不先过正确性关卡,看起来会像"性能胜出",但那是假胜出。
  只有 `check-topk` PASS 之后,SWAR 版的计时数字才可以写进 `RESULTS.md` 的
  popcount 对比表;若 FAIL,先修 kernel,不要记录它的计时。
- 记录两版(builtin / SWAR)的 scan 耗时与有效带宽差异。
- 同时把 Task 3 Step 3 反汇编步骤得到的"是 `cpop` 硬件指令还是 `call`/`jal`
  库函数调用"结论填进 `RESULTS.md` 该表下方的"反汇编观察"一行——这决定了
  两个实现差异的成因解释,不只是记两个数字。

**测完后恢复 builtin 版(若 builtin 更快,或作为默认发布配置):**
```bash
cd hamming_mu && sed -i '/-DPOPCNT_SWAR/d' mu_kernel/CMakeLists.txt && ./build.sh && cd -
```
恢复后建议重跑一次 Task 4 Step 3 的 `check --ref ... --dists ...` 逐位比对,
确认 `sed -i` 撤销没有把 `CMakeLists.txt` 改坏(比如误删了原有别的编译选项)。

**产出去向:** 填入 `RESULTS.md` "popcount 实现对比"表(builtin 与 SWAR 两行的
`scan (ms)`、`有效带宽 (GB/s)`、`正确性` 三列)与其下方的反汇编观察一行。

---

### Task 6 Step 2: 编译 hamming_scan_hist / hamming_merge_topk kernel

**Command:**
```bash
cd hamming_mu && ./build.sh
```

**Expected Output:**
- 编译无错误,无栈相关警告
- `hamming_mu/mu_kernel/mu_kernel.mubin` 时间戳更新(确认是本次重新生成,不是旧产物)
- `hamming_scan_hist` 与 `hamming_merge_topk` 两个符号都出现在产物里(可用
  `$LLVM/bin/llvm-objdump -d hamming_mu/mu_kernel/mu_kernel.mubin | grep -E '<hamming_scan_hist>|<hamming_merge_topk>'` 核实)

**若失败:**
- 若报 `half_float 未声明` 之类错误,说明误加了 half 相关代码 —— 本项目为纯整数路径,`mu_hamming.cpp` 只应 `#include "mu/mu.hpp"`。
- 若报参数个数超限(kernel 参数 <= 9):`hamming_scan_hist` 8 个、`hamming_merge_topk` 8 个,均在限内;若编译器报超限,先检查是否误传了额外参数,而不是删减已有参数。
- 若报候选/直方图索引越界或结果与 `hamming_mu/model_kernel.py` 的 `scan_hist`/`merge_topk` 不一致:以 model_kernel.py 为准逐行比对,尤其是 `cand[2j]=id, cand[2j+1]=dist` 的交错存储、`candCap*2` 的分片大小、以及 merge 里 `dist <= exactThresh` 的过滤条件 —— 这三处是已复现过的静默错误点。
- 若报栈溢出相关警告:`hamming_merge_topk` 唯一的大局部数组是 `uint32_t total[BITS+1]`(257*4=1028 B,远低于 64 KB/task 栈限制),先确认没有其他新增的局部数组吃掉了栈空间。

---

### Step 2: 运行环境校验

**Command:**
```bash
bash hamming_mu/env_check.sh
```

**Expected Output:**
- PCI 设备存在 (lspci -d 20a6: 找到 XCENA 设备)
- dkms status 有匹配当前 uname -r 的 mx_dma 条目
- daxctl list 显示 "mode":"devdax"
- computable=Yes
- validate_host.sh 无 FAIL

**若失败:**
若 mode 是 system-ram,执行:
```bash
sudo daxctl reconfigure-device --mode=devdax --force dax0.0
```
(`--force` 必需:system-ram 模式下该内存已在线挂成 NUMA 节点,必须先下线。若 offline 失败,说明有进程已从该节点分配内存,用 `numactl -H` 查。)

此步不通过不要继续后面任何任务。

---

### Step 6: 编译,确认产出 .mubin 与可执行文件

**Command:**
```bash
chmod +x hamming_mu/build.sh hamming_mu/env_check.sh
cd hamming_mu && ./build.sh
```

**Expected Output:**
- 编译无错误
- `ls -la hamming_mu/mu_kernel/mu_kernel.mubin` 存在
- `ls -la hamming_mu/hamming_scan` 存在

**若失败:**
若报 `half_float 未声明` 之类错误,说明误加了 half 相关代码 —— 本项目为纯整数路径,不应包含 `mu/vector/half.hpp`。

---

### Task 3 Step 2: 编译 hamming_scan_dists kernel

**Command:**
```bash
cd hamming_mu && ./build.sh
```

**Expected Output:**
- 编译无错误
- `hamming_mu/mu_kernel/mu_kernel.mubin` 时间戳更新(确认是本次重新生成,不是旧产物)

**若失败:**
- 若报 `half_float 未声明` 之类错误,说明误加了 half 相关代码 —— 本项目为纯整数路径,`mu_hamming.cpp` 只应 `#include "mu/mu.hpp"`。
- 若要构建 SWAR 变体做对比,加 `-DPOPCNT_SWAR`(需要临时改 `mu_kernel/CMakeLists.txt` 的 `compile_options` 或在 build.sh 里传参;Task 8 的对比会用到这两个变体)。

---

### Task 3 Step 3: 反汇编确认 popcount 是否映射到硬件指令

**Command:**
```bash
source /usr/local/mu_library/mu/script/min_llvm_version_env.sh
LLVM=/usr/local/mu_library/mu_llvm/$XCENA_LLVM_VERSION/$MU_REVISION
$LLVM/bin/llvm-objdump -d hamming_mu/mu_kernel/mu_kernel.mubin \
  | grep -A40 '<hamming_scan_dists>' | grep -iE 'cpop|call|jal' | head -20
```

**Expected Output / 如何解读:**
- 出现 `cpop` 指令 → `__builtin_popcountll` 被映射到了 RISC-V Zbb 扩展的硬件 popcount 指令 —— 默认(非 SWAR)变体应该更快或至少不慢于 SWAR。
- 出现 `call`/`jal` 跳到某个 `__popcount`/`__popcountdi2` 之类的库函数 → 说明 builtin 退化成了软件实现(常见于目标 CPU 未声明 Zbb 扩展),此时手写的 SWAR 变体(`-DPOPCNT_SWAR`)很可能更快,因为它是内联的位运算而非函数调用。
- **把观察到的是 `cpop` 还是 `call`/`jal`,以及具体指令文本,记录到本文件或 Task 8 的报告里** —— Task 8 的 popcount 对比任务需要这个结论来决定要不要以 SWAR 作为默认实现。

---

### Task 4 Step 2: 编译 host 驱动 hamming_scan

**Command:**
```bash
cd hamming_mu && ./build.sh
```

**Expected Output:**
- 编译无错误
- `ls -la hamming_mu/hamming_scan` 存在且为本次新产物(时间戳更新)
- `ls -la hamming_mu/mu_kernel/mu_kernel.mubin` 存在(Task 3 的 kernel 一并编译)

**若失败:**
- 报参数不匹配 / 找不到符号:确认 `hamming_scan.cpp` 里 `module->createFunction("hamming_scan_dists")` 的名字与 `mu_kernel/mu_hamming.cpp` 里 `MU_KERNEL_ADD(hamming_scan_dists)` 完全一致,且 `scanExec->execute(sigs, query, numSigs, dists)` 的参数顺序与个数与 kernel 签名 `(const uint64_t* sigs, const uint64_t* query, uint64_t numSigs, int32_t* outDists)` 一致。
- 报 pxl 相关类型 / 返回值不匹配:对照 `dran_mu/dran_retrieve.cpp`(已知可用的同类 host 驱动)核实 `pxl::MemoryStatus`、`pxl::Result` 的用法,不要自行猜测 API。

---

### Task 4 Step 3: 小规模跑,与 numpy 逐位比对(本项目第一个真正的正确性关卡)

**Command:**
```bash
python hamming_mu/verify_host.py gen -n 10000 -o /tmp/sig10k.bin
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

1. **结果错 + 每次运行结果不同 + 主机自查(host 侧读回来看)正常** → 三条症状同时出现,就是 `flushHostCache` 问题,先查 flush 调用位置(写完数据后、preload/kernel 执行前;kernel 执行后、host 读结果前 各一次)。**不要**去怀疑算法或编译器。
2. 只错部分且稳定复现(每次运行结果一样但就是不对)→ 看 kernel 的 `hostPrintf` 原始比特(hex)输出,与 host 侧同一个值的 hex 对比。浮点/十进制格式化会掩盖差异,原始比特不会。
3. 改了代码但行为没变(像是没生效)→ 加一行新的 `hostPrintf` 看它有没有出现,确认真的重新编译并重传了(不是跑到了旧的 `.mubin` 或旧的可执行文件)。

**待办(需服务器执行后回填):** 实际 PASS/FAIL 输出、rep 计时、有效带宽数值。

---

### Task 5 Step 2: 扫 batchSize 建立基线

**CRITICAL:** 此步必须在任何后续性能测量之前执行。`setBatchSize` 默认为 16,导致 `buildMap(fn, 2816)` 仅激活 `ceil(2816/16) = 176` 个核,其余 2640 个闲置 —— **16 倍性能差异**。若跳过或延后此步,后续所有测量都建立在错误的基线上。

**Command:**
```bash
cd hamming_mu && chmod +x sweep.sh && ./sweep.sh 10000000 2816
```

**Expected Output:**
- 6 行 batchSize 结果(b = 1, 2, 3, 4, 8, 16)
- 预期形状是**中间有谷**: b=1 因分派开销放大而慢,b=16 因并行度不足而慢,最优通常在 **b=3~4**
- 若最优就是 16,说明本负载的分派开销特征与 DRAN 参考项目不同 —— **记录下来,不要强行套用 DRAN 的结论**

**若失败:**
- 若全是 `FAIL`,检查 grep 模式是否与 `hamming_scan.cpp` 第 200 行的输出格式一致
- 若输出不是数字而是标点符号(`:`),说明 awk 字段索引错误 —— 应为 `$6` 而非 `$5`(参考 hamming_scan.cpp 第 200 行的输出格式分析)

---

### Task 5 Step 3: 用最优 batchSize 扫 numSub

**Command:**
```bash
cd hamming_mu && BEST_B=<上一步的最优值> ./sweep.sh 10000000 2816
```

**Expected Output:**
- 4 行 numSub 结果(s = 4, 8, 16, 22)
- 通常在 s=8~16 附近有较平缓的区域

---

### Task 5 Step 4: 记录结果

**Command:**
```bash
编辑 hamming_mu/RESULTS.md,填入上述两个扫描的数字
```

**Expected Output:**
- batchSize 表格中的 6 个测量值和最优值
- numSub 表格中的 4 个测量值和最优值
- 可选:计算有效带宽 = (scannedGB) / (min_ms / 1000) GB/s,其中 scannedGB 在 hamming_scan.cpp 输出中显示

**待办(需服务器执行后回填):** batchSize 和 numSub 的实测值与最优值选择。

---

### Task 7 Step 2: 编译 host 驱动的 C 模式(hamming_scan_hist / hamming_merge_topk 调用路径)

**Command:**
```bash
cd hamming_mu && ./build.sh
```

**Expected Output:**
- 编译无错误
- `ls -la hamming_mu/hamming_scan` 存在且为本次新产物(时间戳更新)
- `ls -la hamming_mu/mu_kernel/mu_kernel.mubin` 存在(Task 6 的 kernel,未改动)

**若失败:**
- 报参数不匹配 / 找不到符号:确认 `hamming_scan.cpp` 里 `module->createFunction("hamming_scan_hist")` / `("hamming_merge_topk")` 的名字与 `mu_kernel/mu_hamming.cpp` 里 `MU_KERNEL_ADD(...)` 完全一致,且 `histExec->execute(...)` / `mergeExec->execute(...)` 的参数顺序、个数与类型与 kernel 签名一致 —— `hamming_scan_hist(sigs, query, numSigs, coarseThresh, candCap, outHist, outCand, outMeta)` 8 个参数,`hamming_merge_topk(hist, cand, meta, taskCount, candCap, topK, dists, outIds)` 8 个参数。
- 报 pxl 相关类型 / 返回值不匹配:对照 `dran_mu/dran_retrieve.cpp`(已知可用的同类 host 驱动)核实 `pxl::MemoryStatus`、`pxl::Result`、`job->buildMap`、`execute`/`synchronize` 的用法,不要自行猜测 API。
- 报 signed/unsigned 比较警告:`ids[topK + 2] < (uint32_t)topK` 之类的比较里 `topK` 已显式转 `uint32_t`,若警告指向别处,先看是不是新增代码里漏转了。

---

### Task 7 Step 3: 小规模验证 top-k 正确(mode C 端到端)

**Command:**
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 64 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/dev10k.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/dev10k.ids -k 100
```

（若 `/tmp/sig10k.bin` 和 `.query`/`.dists.npy` 还不存在，先用 Task 4 的
`python hamming_mu/verify_host.py gen -n 10000 -o /tmp/sig10k.bin` 生成。）

**Expected Output:**
```
PASS: 100 个 id 距离全部 <= 第k名(...)
```

**如何解读 / 平局语义:** 判据是"返回的每个 id 的距离必须 <= 第 k 名的距离"，
而**不是** id 集合与 numpy argsort 完全相同——计数排序与 numpy 在平局时选的
具体 id 可以不同，这不是错误。`check-topk` 同时检查数量、距离上界、id 唯一性
三项；三项皆过才算 PASS。

**若失败,按此顺序诊断:**
1. 数量不足(collected < k)但没有触发重试/警告 → 检查重试循环里
   `ids[topK]` / `ids[topK+1]` / `ids[topK+2]` 三个状态槽的索引是否与
   `mu_hamming.cpp` 里 `outIds[topK]=exactThresh`、`outIds[topK+1]=anyOverflow`、
   `outIds[topK+2]=collected` 完全对应——错位是最容易犯且最隐蔽的错误。
2. 结果每次运行不同 → 读 `ids` 之前是否每次都调用了
   `pxl::flushHostCache(ids, (topK+3)*sizeof(uint32_t))`；重试循环内、循环外、
   dump 之前都需要各自的 flush，漏一处就会读到设备写入前的陈旧数据。
3. 距离超过第 k 名 → 检查 `hamming_merge_topk` 里 `dist <= exactThresh` 的过滤
   条件是否被误删或改动（这是 Task 2b 复现过的静默错误点，host 侧不应改动
   kernel，但要确认 host 传给 kernel 的 `topK`/`candCap`/`taskCount` 参数顺序
   没有传错导致 kernel 侧算出错误的 exactThresh）。

**待办(需服务器执行后回填):** 实际 PASS/FAIL 输出、scan/merge 分别计时、
是否触发过重试。

---

### Task 7 Step 4: 验证候选溢出路径真的会触发并自愈

**Command:**
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 4 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/ovf.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/ovf.ids -k 100
```

（只用 4 个 task，每 task 要扫 2500 个签名，候选数远超 `candCap = topK*4 = 400`，
必然触发 kernel 侧的 `overflow=1`。）

**Expected Output:**
- kernel 端打印 `[kernel] task0 ... overflow=1`
- 最终 `check-topk` **PASS**
- host 侧输出以下两种之一，都是正确行为：
  - `注:发生过候选溢出,但结果仍有效(全部 d <= 精确阈值 N)` ——
    溢出了但仍收集够 k 个；merge 已按 exactThresh 过滤，这些 id 全部合格，无需重跑。
  - `收集不足(x/100)-> coarseThresh 118 -> 精确阈值 N 重跑` ——
    溢出丢掉了太多合格项，重跑收敛到精确阈值后应能收集够。

**关键判据是 `check-topk` PASS**，而不是"有没有发生溢出"——溢出本身是预期
会触发的诊断信息，不是失败信号。

**若失败:**
- 若看到 `已在精确阈值 N 仍只收集到 x/100 -> candCap 太小`：说明重试已经收敛
  到精确阈值但 `candCap` 仍不够大，需要在 host 侧把 `candCap`（当前
  `topK*4`）调大后重跑，重试循环本身不会再有帮助——这是设计上的"重试无用"
  分支，不是 bug。
- 若 `check-topk` FAIL 但没有任何"收集不足"或"候选溢出"提示：先确认
  `-t 4` 确实生效（看 host 打印的 `tasks=4`），不是意外用了默认 taskCount。
- 若 5 次重试后仍未收敛（`for (int attempt = 0; attempt < 5; ...)` 用尽）：
  记录 `coarseThresh` 的变化轨迹，检查是否在两个 `exactThresh` 值之间震荡而非
  收敛——按设计它应该单调收敛到某个不动点。

**待办(需服务器执行后回填):** 是否观察到 overflow、host 侧走了哪条分支、
`check-topk` 的最终 PASS/FAIL、以及若调大 candCap 重试过的话新的结果。
