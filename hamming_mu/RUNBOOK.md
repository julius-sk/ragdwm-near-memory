# RUNBOOK

## 待服务器验证

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
