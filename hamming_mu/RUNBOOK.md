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
