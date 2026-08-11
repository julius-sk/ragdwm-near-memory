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
