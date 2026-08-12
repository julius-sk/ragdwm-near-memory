# MX1P 近内存 Hamming 扫描(XOR + POPCOUNT)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 XCENA MX1P 上实现 256 位签名的 Hamming 距离扫描与 top-k 选择,产出"近内存 vs host"的三组可信对比数字,用于 DWM/RAGDWM 论文的硬件章节。

**Architecture:** 两个 MU kernel —— `scan_hamming` 分条扫描签名(每 task 一段连续内存,XOR+POPCOUNT 得 0~256 距离,累加 257 桶直方图并收集候选),`merge_topk` 单 task 归并直方图定出精确距离阈值并收集 k 个 id。主机侧单 arena + flushHostCache + 命令行旋钮。三组对比:A=host numba 基线,B=设备扫描回传全部距离,C=设备扫描+设备 top-k。

**Tech Stack:** XCENA SDK v1.4.9、PXL C++ host API、MU LLVM RISC-V 工具链(riscv64 / metisx-d)、CMake + Ninja、Python 3 + numpy(参考实现与比对)。

## Global Constraints

- **平台**:XCENA MX1P,CXL 3.2 Type 3;固件必须 Computing 模式(`xcena_cli device-info` 显示 `computable=Yes`);DAX 必须 devdax 模式。
- **kernel 硬约束**:参数 ≤ 9 个;每 task 堆 3 MB;每 task 栈 **64 KB**(局部数组必须算清楚);入口用 `MU_KERNEL_ADD(fn_name)`,名字须与 host 端 `createFunction` 完全一致。
- **头文件**:`mu.hpp` 不包含 `vector.hpp` / `half.hpp`,需要时必须显式 include。本项目为纯整数路径,只需 `mu/mu.hpp`。
- **preload 粒度**:1 GB 对齐 + 1 GB 整数倍。全项目只用**一个 arena**,所有 buffer 按偏移切入,整体 preload 一次。
- **flushHostCache 是正确性的一部分,不是优化**:主机写完、kernel 启动前必须 flush;kernel 跑完、主机读结果前必须 flush。
- **签名规格**:256 位 = 4 × `uint64_t` = 32 字节,每签名连续存放,签名 i 占 `[i*32, i*32+32)`。
- **距离值域**:Hamming 距离 ∈ [0, 256],共 **257** 个桶。
- **编译选项**(照抄 dran_mu):`--target=riscv64 -mcpu=metisx-d -ffreestanding -nostdlib -nodefaultlibs -fno-exceptions -Werror=return-type -g -O3 -std=c++17 -fno-rtti -ffunction-sections -fdata-sections`
- **测量口径**:稳态每查询延迟(数据已在卡上,不含一次性 fill/flush/preload);初始化成本单独报告。每配置跑 ≥5 次取**最小值**,计时前先 warmup 一轮。
- **正确性优先**:任何性能数字之前,设备结果必须与 numpy 参考**逐位一致**。DRAN 项目中手写标量循环曾在同一工具链 -O3 下产生错误结果,故两种 popcount 实现都必须先验证。
- **不改动 `dran_mu/` 下任何文件。**

## 执行环境划分(重要)

**开发机(本地)没有 XCENA 硬件,也没有任何 C++ 编译器**(g++/clang/cmake/ninja 均无),
只有 Python 3 + numpy 2.4.6(`bitwise_count` 可用)。因此每一步都标注了执行位置:

| 标记 | 含义 | 谁执行 |
|---|---|---|
| **[本地]** | 写代码、跑 Python 测试 | subagent 直接做,并验证 |
| **[服务器]** | 需要编译或 XCENA 硬件 | **subagent 不执行**,写进 `hamming_mu/RUNBOOK.md` 交给人去跑 |

**subagent 规则:遇到 [服务器] 步骤,不要尝试执行,也不要伪造结果。**
把命令与预期输出追加进 `hamming_mu/RUNBOOK.md`,标记该步为"待服务器验证",然后继续下一步。

**风险控制**:C++ 无法本地编译,但 kernel 的**算法**(257 桶直方图 → 精确阈值 →
候选收集 → 溢出重试)是纯逻辑,Task 2b 用 Python 精确建模并对 numpy 模糊测试。
算法在本地验对之后,服务器上只剩环境与性能问题,不必再调算法。

---

### Task 1: 环境校验与目录骨架

**Files:**
- Create: `hamming_mu/build.sh`
- Create: `hamming_mu/CMakeLists.txt`
- Create: `hamming_mu/mu_kernel/CMakeLists.txt`
- Create: `hamming_mu/mu_kernel/mu_hamming.cpp`
- Create: `hamming_mu/hamming_scan.cpp`
- Create: `hamming_mu/env_check.sh`

**Interfaces:**
- Consumes: 无(首个任务)
- Produces: 可编译通过的空骨架;`hamming_mu/mu_kernel/mu_kernel.mubin` 产物路径;kernel 符号 `hamming_scan`(此时为空实现)

- [ ] **Step 1: 写环境校验脚本**

创建 `hamming_mu/env_check.sh`:

```bash
#!/bin/bash
# 环境校验 —— 任何调试之前先跑这个,不要手工猜断在哪一环。
set -u
SDK="${1:-$HOME/engramme/xcena_sdk/xcena_sdk_v1.4.9}"

echo "=== 1. PCI 设备 (vendor 20a6) ==="
lspci -d 20a6: || echo "  !! 没有找到 XCENA 设备"

echo "=== 2. mx_dma DKMS 驱动 (须匹配当前内核) ==="
echo "  当前内核: $(uname -r)"
dkms status | grep -i mx_dma || echo "  !! mx_dma 无 DKMS 条目 —— 换过内核?需重建"
ls /dev/mx_dma/ 2>/dev/null || echo "  !! /dev/mx_dma/ 不存在"

echo "=== 3. DAX 模式 (必须是 devdax,不跨重启保留) ==="
daxctl list || echo "  !! daxctl 不可用"
echo "  如果 mode 不是 devdax:"
echo "    sudo daxctl reconfigure-device --mode=devdax --force dax0.0"

echo "=== 4. 固件 Computing 模式 (须 computable=Yes) ==="
xcena_cli device-info 2>/dev/null | grep -i computable || echo "  !! 无法读取 computable"

echo "=== 5. SDK 自带完整校验 ==="
if [ -f "${SDK}/scripts/validate_host.sh" ]; then
    bash "${SDK}/scripts/validate_host.sh" 2>&1 | tail -30
else
    echo "  !! 未找到 ${SDK}/scripts/validate_host.sh"
fi
```

- [ ] **Step 2: 运行环境校验,确认五项全过**

Run: `bash hamming_mu/env_check.sh`

Expected: PCI 设备存在;`dkms status` 有匹配当前 `uname -r` 的 mx_dma 条目;`daxctl list` 显示 `"mode":"devdax"`;`computable=Yes`;`validate_host.sh` 无 FAIL。

若 `mode` 是 `system-ram`,执行:
```bash
sudo daxctl reconfigure-device --mode=devdax --force dax0.0
```
(`--force` 必需:system-ram 模式下该内存已在线挂成 NUMA 节点,必须先下线。若 offline 失败,说明有进程已从该节点分配内存,用 `numactl -H` 查。)

**此步不通过不要继续后面任何任务。**

- [ ] **Step 3: 写 kernel 空骨架**

创建 `hamming_mu/mu_kernel/mu_hamming.cpp`:

```cpp
// mu_hamming.cpp — 256 位签名的 Hamming 扫描 kernel(XCENA MX1P)。
//
// 负载:query 签名对 N 个存储签名做 XOR + POPCOUNT,得 0..256 的距离,选 top-k。
// 每签名 4 x uint64 = 32B 连续存放,每 task 扫一段连续签名 —— DRAM 最擅长的顺序读。
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

// 空骨架:仅验证工具链能编译并产出 .mubin。
void hamming_scan(const uint64_t* sigs, uint64_t numSigs, uint32_t* out)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    if (taskIdx == 0)
    {
        mu::hostPrintf("[kernel] skeleton alive: numSigs=%u\n", (unsigned)numSigs);
        out[0] = 1u;
    }
}

MU_KERNEL_ADD(hamming_scan);
```

- [ ] **Step 4: 写 kernel 侧 CMakeLists**

创建 `hamming_mu/mu_kernel/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.11)
project(mu_kernel)

set(CMAKE_C_COMPILER ${MU_LLVM_PATH}/bin/clang)
set(CMAKE_CXX_COMPILER ${MU_LLVM_PATH}/bin/clang)

set(output mu_kernel.mubin)
set(output_path ${CMAKE_CURRENT_SOURCE_DIR})

set(include_directories
    ${MU_LLVM_PATH}/picolibc-rv
    ${MU_LLVM_PATH}/picolibc-rv/include
    ${MU_LLVM_PATH}/libcxx-rv/include/c++/v1
    ${MU_LIB_PATH}/include
)

set(link_directories
    ${MU_LLVM_PATH}/builtins-rv/lib/linux
    ${MU_LLVM_PATH}/libcxx-rv/lib
    ${MU_LLVM_PATH}/libcxxabi-rv/lib
    ${MU_LLVM_PATH}/picolibc-rv/newlib
    ${MU_LIB_PATH}/lib
)

set(compile_options
    --target=riscv64
    -mcpu=metisx-d
    -ffreestanding
    -nostdlib
    -nodefaultlibs
    -fno-exceptions
    -Werror=return-type
    -g
    -O3
    -std=c++17
    -fno-rtti
    -ffunction-sections
    -fdata-sections
)

set(link_options
    -lclang_rt.builtins-riscv64
    -z max-page-size=4
    -lc
    -lc++
    -lc++abi
    -Wl,--gc-sections
    --ld-path=${MU_LLVM_PATH}/bin/ld.lld
)

set(start_up ${MU_LIB_PATH}/script/mu_startup.s)
set(linker ${MU_LIB_PATH}/script/mu_linker.ld)

set(src
    mu_hamming.cpp
)

add_executable(${output} ${src})

target_include_directories(${output} PRIVATE ${include_directories})
target_link_libraries(${output} PRIVATE mu_std)
target_link_directories(${output} PRIVATE ${link_directories})
target_compile_options(${output} PRIVATE ${compile_options})
target_link_options(${output} PRIVATE ${compile_options} ${link_options} ${start_up} ${linker})
set_target_properties(${output} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${output_path})
install(TARGETS ${output} RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/mu_kernel)
```

- [ ] **Step 5: 写 host 空骨架与顶层 CMakeLists**

创建 `hamming_mu/hamming_scan.cpp`:

```cpp
// hamming_scan.cpp — host 驱动:MX1P 上的 Hamming 扫描与 top-k。
//
// Build: 见 build.sh(需 XCENA SDK v1.4.9、PXL、MU LLVM 工具链)

#include <cstdio>
#include "pxl/pxl.hpp"

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;
    auto context = pxl::createContext(0);
    if (!context)
    {
        printf("createContext failed\n");
        return 1;
    }
    printf("skeleton: context created\n");
    pxl::destroyContext(context);
    return 0;
}
```

创建 `hamming_mu/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.11)
project(hamming_scan)

set(src
    hamming_scan.cpp
)

set(include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(shared_libs)

if (NOT XCENA_SDK_CMAKE_BUILD)
    find_package(pxl REQUIRED)
    message(STATUS "PXL library found at: ${pxl_DIR}")
    LIST(APPEND shared_libs pxl::pxl)
else()
    LIST(APPEND shared_libs pxl)
endif()

foreach(file ${src})
    get_filename_component(target ${file} NAME_WE)
    add_executable(${target} ${file})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_include_directories(${target} PRIVATE ${include})
    target_link_libraries(${target} ${shared_libs})
    install(TARGETS ${target} RUNTIME DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

include(ExternalProject)
ExternalProject_Add(build_mu_${PROJECT_NAME}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/mu_kernel
    BUILD_ALWAYS TRUE
    CMAKE_ARGS
        -DMU_LIB_PATH=$ENV{MU_LIB_PATH}
        -DMU_LLVM_PATH=$ENV{MU_LLVM_PATH}
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}
)
```

创建 `hamming_mu/build.sh`:

```bash
#!/bin/bash
set -e

export MU_LIB_PATH=/usr/local/mu_library/mu

source ${MU_LIB_PATH}/script/min_llvm_version_env.sh
if [ -z "${XCENA_LLVM_VERSION}" ] || [ -z "${MU_REVISION}" ]; then
    echo "XCENA_LLVM_VERSION or MU_REVISION is not set in min_llvm_version_env.sh"
    exit 1
fi
export MU_LLVM_PATH=/usr/local/mu_library/mu_llvm/$XCENA_LLVM_VERSION/$MU_REVISION/

rm -rf build
mkdir -p build
cd build
/usr/bin/cmake .. -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
ninja
ninja install
cd -
```

- [ ] **Step 6: 编译,确认产出 .mubin 与可执行文件**

Run:
```bash
chmod +x hamming_mu/build.sh hamming_mu/env_check.sh
cd hamming_mu && ./build.sh
```

Expected: 编译无错误;`ls -la hamming_mu/mu_kernel/mu_kernel.mubin` 存在;`ls -la hamming_mu/hamming_scan` 存在。

若报 `half_float 未声明` 之类错误,说明误加了 half 相关代码 —— 本项目为纯整数路径,不应包含 `mu/vector/half.hpp`。

- [ ] **Step 7: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/
git commit -m "feat(hamming): scaffold MX1P Hamming scan project"
```

(若 `engramme` 尚未初始化为 git 仓库,先 `git init` 并确认 `.gitignore` 排除 `build/` 与 `*.mubin`。)

---

### Task 2: numpy 参考实现与测试数据生成

**Files:**
- Create: `hamming_mu/verify_host.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `gen_sigs(n, seed) -> np.ndarray`,形状 `(n, 4)`,dtype `uint64`
  - `sigs_to_bytes(sigs) -> bytes`,长度 `n*32`,小端序
  - `hamming_all(sigs, query) -> np.ndarray`,形状 `(n,)`,dtype `int32`,值域 [0,256]
  - `topk_ids(dists, k) -> np.ndarray`,形状 `(k,)`,dtype `uint32`,按 (距离升序, id 升序) 稳定排序
  - CLI:`python verify_host.py gen -n N -o FILE`、`python verify_host.py check -n N --dists FILE`、`python verify_host.py check-topk -n N -k K --ids FILE`

- [ ] **Step 1: 写失败的测试**

创建 `hamming_mu/verify_host.py`,先只写测试部分:

```python
"""Hamming 扫描的 numpy 参考实现 + 设备输出比对工具。

设备侧签名布局:每签名 4 x uint64 = 32B 连续,小端序。
本文件既是参考实现,也是 CLI 工具(gen / check / check-topk)。
"""
import argparse
import sys

import numpy as np

WORDS = 4                 # 每签名 4 个 uint64
BITS = WORDS * 64         # 256 位
NBUCKETS = BITS + 1       # 距离 0..256,共 257 个桶


def _self_test():
    # 1. 全 0 与全 1 的距离必须是 256
    a = np.zeros((1, WORDS), dtype=np.uint64)
    b = np.full((1, WORDS), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    assert hamming_all(b, a[0])[0] == 256, "全0 vs 全1 应为 256"

    # 2. 自己和自己的距离必须是 0
    s = gen_sigs(16, seed=1)
    assert hamming_all(s, s[0])[0] == 0, "自距离应为 0"

    # 3. 值域必须落在 [0,256]
    d = hamming_all(s, s[3])
    assert d.min() >= 0 and d.max() <= BITS, "距离越界"

    # 4. 单比特差异的距离必须是 1
    c = s[0].copy()
    c[0] = np.uint64(c[0] ^ np.uint64(1))
    assert hamming_all(c.reshape(1, WORDS), s[0])[0] == 1, "单比特差应为 1"

    # 5. 字节序往返必须无损
    raw = sigs_to_bytes(s)
    back = np.frombuffer(raw, dtype='<u8').reshape(-1, WORDS)
    assert np.array_equal(back, s), "字节序往返不一致"

    # 6. top-k 必须按 (距离, id) 稳定排序
    dd = np.array([3, 1, 1, 0, 5], dtype=np.int32)
    assert list(topk_ids(dd, 3)) == [3, 1, 2], "top-k 平局需按 id 升序"

    print("self-test: ALL PASS")


if __name__ == "__main__":
    _self_test()
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `python hamming_mu/verify_host.py`

Expected: FAIL —— `NameError: name 'hamming_all' is not defined`

- [ ] **Step 3: 写实现**

在 `hamming_mu/verify_host.py` 中,把下列函数插入到 `_self_test` **之前**:

```python
def gen_sigs(n, seed=0):
    """生成 n 个随机 256 位签名,形状 (n,4) 的 uint64。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n, WORDS), dtype=np.uint64)


def sigs_to_bytes(sigs):
    """转成设备侧内存布局:每签名 4 x uint64 连续,小端序。"""
    return np.ascontiguousarray(sigs, dtype='<u8').tobytes()


def hamming_all(sigs, query):
    """query 对全部 sigs 的 Hamming 距离,返回 (n,) int32,值域 [0,256]。"""
    sigs = np.asarray(sigs, dtype=np.uint64).reshape(-1, WORDS)
    query = np.asarray(query, dtype=np.uint64).reshape(WORDS)
    x = np.bitwise_xor(sigs, query)
    # 逐字节查表 popcount:对 uint64 视图按字节展开再累加
    b = x.view(np.uint8).reshape(-1, WORDS * 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int32)


def topk_ids(dists, k):
    """最近的 k 个 id,按 (距离升序, id 升序) 稳定排序,返回 (k,) uint32。"""
    dists = np.asarray(dists)
    order = np.argsort(dists, kind="stable")   # stable => 平局按 id 升序
    return order[:k].astype(np.uint32)
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `python hamming_mu/verify_host.py`

Expected: `self-test: ALL PASS`

- [ ] **Step 5: 加 CLI(生成数据与比对设备输出)**

把 `if __name__ == "__main__":` 块整体替换为:

```python
def _cmd_gen(args):
    sigs = gen_sigs(args.n, seed=args.seed)
    query = gen_sigs(1, seed=args.seed + 12345)[0]
    with open(args.out, "wb") as f:
        f.write(sigs_to_bytes(sigs))
    with open(args.out + ".query", "wb") as f:
        f.write(sigs_to_bytes(query.reshape(1, WORDS)))
    d = hamming_all(sigs, query)
    np.save(args.out + ".dists.npy", d)
    print(f"wrote {args.n} sigs -> {args.out} ({args.n*32} B)")
    print(f"  dist min={d.min()} max={d.max()} mean={d.mean():.2f}")


def _cmd_check(args):
    ref = np.load(args.ref)
    got = np.fromfile(args.dists, dtype=np.int32)
    if got.size != ref.size:
        print(f"FAIL: 长度不符 ref={ref.size} got={got.size}")
        sys.exit(1)
    bad = np.nonzero(got != ref)[0]
    if bad.size:
        i = bad[0]
        print(f"FAIL: {bad.size} 处不符,首个 idx={i} ref={ref[i]} got={got[i]}")
        sys.exit(1)
    print(f"PASS: {ref.size} 个距离逐位一致")


def _cmd_check_topk(args):
    ref = np.load(args.ref)
    want = set(topk_ids(ref, args.k).tolist())
    got = np.fromfile(args.ids, dtype=np.uint32)
    if got.size != args.k:
        print(f"FAIL: 期望 {args.k} 个 id,得到 {got.size}")
        sys.exit(1)
    # 平局可能使具体 id 不同,但每个返回 id 的距离必须 <= 第 k 名的距离
    kth = np.sort(ref)[args.k - 1]
    biggest = ref[got].max()
    if biggest > kth:
        print(f"FAIL: 返回了距离 {biggest} 的 id,但第 {args.k} 名距离是 {kth}")
        sys.exit(1)
    overlap = len(want & set(got.tolist()))
    print(f"PASS: {args.k} 个 id 距离全部 <= 第k名({kth});与参考重合 {overlap}/{args.k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen"); g.set_defaults(fn=_cmd_gen)
    g.add_argument("-n", type=int, required=True)
    g.add_argument("-o", required=True)
    g.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("check"); c.set_defaults(fn=_cmd_check)
    c.add_argument("--ref", required=True)
    c.add_argument("--dists", required=True)

    t = sub.add_parser("check-topk"); t.set_defaults(fn=_cmd_check_topk)
    t.add_argument("--ref", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("-k", type=int, required=True)

    s = sub.add_parser("selftest"); s.set_defaults(fn=lambda a: _self_test())

    args = p.parse_args()
    args.fn(args)
```

- [ ] **Step 6: 验证 CLI 可用**

Run:
```bash
python hamming_mu/verify_host.py selftest
python hamming_mu/verify_host.py gen -n 10000 -o /tmp/sig10k.bin
```

Expected: `self-test: ALL PASS`;输出 `wrote 10000 sigs -> /tmp/sig10k.bin (320000 B)` 及距离统计(mean 应接近 128,因为随机签名期望距离为 BITS/2)。

- [ ] **Step 7: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/verify_host.py
git commit -m "feat(hamming): numpy reference + data generator with self-tests"
```

---

### Task 2b: kernel 算法的 Python 模型与模糊测试 [本地]

**Files:**
- Create: `hamming_mu/model_kernel.py`

**Interfaces:**
- Consumes: Task 2 的 `gen_sigs` / `hamming_all` / `topk_ids`
- Produces:
  - `scan_hist(sigs, query, task_count, coarse_thresh, cand_cap) -> (hist, cand, meta)`
    —— 精确模拟 `hamming_scan_hist`,`hist` 形状 `(task_count, 257)`,
    `cand` 形状 `(task_count, cand_cap * 2)`(交错 `(id, dist)`,`cand_cap` 计**条目数**),
    `meta` 形状 `(task_count, 2)` = [candCount, overflow]
  - `merge_topk(hist, cand, meta, top_k) -> (ids, exact_thresh, any_overflow, collected)`
    —— 精确模拟 `hamming_merge_topk`
  - `run_with_retry(sigs, query, task_count, top_k, cand_cap, thresh0) -> (ids, n_attempts)`
    —— 精确模拟 host 端的溢出重试循环

**为什么需要这个任务:** C++ 无法本地编译,但上述三段是整个计划里唯一
"想错了会静默出错"的逻辑。用 Python 逐行对应地实现一遍并模糊测试,
可以在不碰硬件的前提下证明算法本身正确。**这不是替代设备测试,
而是把算法 bug 与环境 bug 分离**,避免在服务器上同时调两件事。

- [ ] **Step 1: 写失败的测试 [本地]**

创建 `hamming_mu/model_kernel.py`:

```python
"""kernel 算法的 Python 模型 —— 与 mu_hamming.cpp 逐行对应。

用途:本地(无 XCENA、无 C++ 编译器)验证计数排序 top-k 与溢出重试的正确性。
模型改了,C++ 也要改;两边逻辑必须保持一致。
"""
import numpy as np

from verify_host import BITS, NBUCKETS, WORDS, gen_sigs, hamming_all, topk_ids


def _task_range(num_sigs, task_idx, task_count):
    """与 kernel 的 myRange 逐行对应。

    分块规则是 **ceil 分块**:每 task 取 ceil(n/taskCount) 个,末尾的 task
    分到的**更少、甚至为空**(不是"最后一个吃掉余数")。两种约定给出不同划分,
    例如 n=10, taskCount=4:
        ceil 分块      -> [0,3) [3,6) [6,9) [9,10)
        余数给最后一个 -> [0,2) [2,4) [4,6) [6,10)
    C++ 侧的 myRange 必须用同一条规则,否则模型与设备会静默分歧。
    """
    per_task = (num_sigs + task_count - 1) // task_count
    begin = task_idx * per_task
    end = min(begin + per_task, num_sigs)
    return min(begin, num_sigs), end


def _fuzz():
    rng = np.random.default_rng(7)
    for trial in range(200):
        n = int(rng.integers(1, 3000))
        tc = int(rng.integers(1, 65))
        k = int(rng.integers(1, min(n, 200) + 1))
        cap = int(rng.integers(1, 300))
        thresh = int(rng.integers(0, BITS + 1))

        sigs = gen_sigs(n, seed=trial)
        query = gen_sigs(1, seed=trial + 9999)[0]
        ref = hamming_all(sigs, query)

        ids, attempts = run_with_retry(sigs, query, tc, k, cap, thresh)

        # 不变式 1:凑够 k 个(除非 cap*tc 物理上装不下 k 个)
        if cap * tc >= k:
            assert len(ids) == k, f"trial {trial}: 只收到 {len(ids)}/{k}"
            # 不变式 2:每个返回 id 的距离必须 <= 第 k 名的距离(平局允许换人)
            kth = np.sort(ref)[k - 1]
            assert ref[ids].max() <= kth, (
                f"trial {trial}: 返回距离 {ref[ids].max()} > 第k名 {kth}")
            # 不变式 3:无重复 id
            assert len(set(ids.tolist())) == len(ids), f"trial {trial}: 有重复 id"
        # 不变式 4:重试次数有界
        assert attempts <= 5, f"trial {trial}: 重试 {attempts} 次未收敛"

    # 定向用例:阈值极紧(收集不足)与极松(必然溢出)都要能自愈
    sigs = gen_sigs(2000, seed=1)
    query = gen_sigs(1, seed=2)[0]
    ref = hamming_all(sigs, query)
    for thresh0 in (0, BITS):
        ids, attempts = run_with_retry(sigs, query, 4, 50, 60, thresh0)
        kth = np.sort(ref)[49]
        assert len(ids) == 50 and ref[ids].max() <= kth, f"thresh0={thresh0} 未自愈"
        assert attempts >= 1, f"thresh0={thresh0} 应触发至少一次重试"

    # 直方图必须统计全部签名,不受 cand_cap 影响(溢出自愈的前提)
    hist, _cand, _meta = scan_hist(sigs, query, 4, coarse_thresh=BITS, cand_cap=1)
    assert hist.sum() == len(sigs), "直方图丢了签名 —— 溢出自愈的前提被破坏"

    # ---- 回归用例:粗阈值远大于精确阈值时的静默错误 ----
    # 初版设计(候选只存 id、merge 不按 exact_thresh 过滤)在此处会
    # "成功门通过但结果错":返回距离 145,而真实第 k 名距离是 113。
    # 这个用例锁死该 bug,任何回退到"不过滤"的改动都会在这里响亮失败。
    sigs = gen_sigs(2000, seed=0)
    query = gen_sigs(1, seed=9999)[0]
    ref = hamming_all(sigs, query)
    k = 50
    kth = int(np.sort(ref)[k - 1])
    for coarse in (BITS, 148, 130, kth):          # 从极松到刚好
        hist, cand, meta = scan_hist(sigs, query, 4, coarse, cand_cap=100000)
        ids, exact_thresh, _ov, collected = merge_topk(hist, cand, meta, k)
        assert exact_thresh == kth, f"coarse={coarse}: 精确阈值 {exact_thresh} != 第k名 {kth}"
        assert collected == k, f"coarse={coarse}: 只收集到 {collected}/{k}"
        assert ref[ids].max() <= kth, (
            f"coarse={coarse}: 返回距离 {ref[ids].max()} > 第k名 {kth} —— "
            "候选未按 exact_thresh 过滤(静默错误回归)")

    print("model fuzz: ALL PASS (200 随机 + 定向用例 + 松阈值回归)")


if __name__ == "__main__":
    _fuzz()
```

- [ ] **Step 2: 运行测试,确认失败 [本地]**

Run: `cd hamming_mu && python model_kernel.py`

Expected: FAIL —— `NameError: name 'run_with_retry' is not defined`

- [ ] **Step 3: 写模型实现 [本地]**

在 `model_kernel.py` 中,把下列函数插入到 `_fuzz` **之前**:

```python
def scan_hist(sigs, query, task_count, coarse_thresh, cand_cap):
    """模拟 hamming_scan_hist:每 task 一份 257 桶直方图 + 候选分片。

    候选交错存放 (id, dist):cand[t, 2j]=id, cand[t, 2j+1]=dist。
    存距离是正确性的必要条件 —— 见 merge_topk 的过滤。
    """
    n = len(sigs)
    dists = hamming_all(sigs, query)
    hist = np.zeros((task_count, NBUCKETS), dtype=np.uint32)
    cand = np.zeros((task_count, cand_cap * 2), dtype=np.uint32)   # 交错 (id, dist)
    meta = np.zeros((task_count, 2), dtype=np.uint32)   # [candCount, overflow]

    for t in range(task_count):
        begin, end = _task_range(n, t, task_count)
        n_cand = 0
        overflow = 0
        for i in range(begin, end):
            d = int(dists[i])
            hist[t, d] += 1                      # 直方图统计全部,不受 cap 影响
            if d <= coarse_thresh:
                if n_cand < cand_cap:
                    cand[t, 2 * n_cand] = i
                    cand[t, 2 * n_cand + 1] = d  # ★ 必须连距离一起存
                    n_cand += 1
                else:
                    overflow = 1                 # 记录但不中断
        meta[t, 0] = n_cand
        meta[t, 1] = overflow
    return hist, cand, meta


def merge_topk(hist, cand, meta, top_k):
    """模拟 hamming_merge_topk:归并直方图定精确阈值,按阈值过滤候选。"""
    task_count = hist.shape[0]
    total = hist.sum(axis=0, dtype=np.uint64)

    exact_thresh = BITS
    cum = 0
    for b in range(NBUCKETS):
        cum += int(total[b])
        if cum >= top_k:
            exact_thresh = b
            break

    any_overflow = int(meta[:, 1].any())

    # ★ 只收 dist <= exact_thresh 的候选。没有这一步,粗阈值偏大时会返回
    #   宽带里的任意 id(已复现的静默错误:返回距离 145 而真实第k名是 113)。
    ids = []
    for t in range(task_count):
        if len(ids) >= top_k:
            break
        cnt = int(meta[t, 0])
        for j in range(cnt):
            if len(ids) >= top_k:
                break
            if int(cand[t, 2 * j + 1]) <= exact_thresh:
                ids.append(int(cand[t, 2 * j]))

    collected = len(ids)
    return np.array(ids, dtype=np.uint32), exact_thresh, any_overflow, collected


def run_with_retry(sigs, query, task_count, top_k, cand_cap, thresh0, max_attempts=5):
    """模拟 host 端的重试循环。

    因为 merge 已按 exact_thresh 过滤,收集到的 id 必然满足 d <= 第k名距离,
    所以**溢出不会产生错误结果**,只可能导致收集不足。重试条件因此只看数量:
      collected >= top_k          -> 完成(即便 overflow=1)
      collected <  top_k          -> 令 thresh = exact_thresh 重跑
                                     (过紧则放宽、过松则收紧,两个方向都收敛)
      已在 exact_thresh 仍不足    -> cand_cap 太小,调用方需增大它
    """
    thresh = thresh0
    attempts = 0
    ids = np.array([], dtype=np.uint32)
    for _ in range(max_attempts):
        hist, cand, meta = scan_hist(sigs, query, task_count, thresh, cand_cap)
        ids, exact_thresh, any_overflow, collected = merge_topk(hist, cand, meta, top_k)
        if collected >= top_k:
            break
        if thresh == exact_thresh:
            break            # 已在精确阈值仍不足 -> cand_cap 受限,重试无用
        thresh = exact_thresh
        attempts += 1
    return ids, attempts
```

- [ ] **Step 4: 运行测试,确认通过 [本地]**

Run: `cd hamming_mu && python model_kernel.py`

Expected: `model fuzz: ALL PASS (200 随机 + 定向用例)`

若某条不变式失败,说明**算法本身**有问题,必须先修模型再同步改 C++ —— 
这正是本任务的目的:在没有硬件的情况下抓出算法 bug。

- [ ] **Step 5: 提交 [本地]**

```bash
cd "$HOME/engramme"
git add hamming_mu/model_kernel.py
git commit -m "test(hamming): python model of kernel algorithm, fuzzed vs numpy"
```

---

### Task 3: B 组 kernel —— 距离扫描(不做 top-k)

**Files:**
- Modify: `hamming_mu/mu_kernel/mu_hamming.cpp`(替换空骨架)

**Interfaces:**
- Consumes: Task 1 的构建系统;Task 2 的 `hamming_all` 作为正确性基准
- Produces: kernel 符号 `hamming_scan_dists`,签名:
  ```cpp
  void hamming_scan_dists(const uint64_t* sigs, const uint64_t* query,
                          uint64_t numSigs, int32_t* outDists)
  ```
  语义:`outDists[i]` = 第 i 个签名与 query 的 Hamming 距离,∈ [0,256]

- [ ] **Step 1: 写 kernel 实现**

把 `hamming_mu/mu_kernel/mu_hamming.cpp` 整体替换为:

```cpp
// mu_hamming.cpp — 256 位签名的 Hamming 扫描 kernel(XCENA MX1P)。
//
// 负载:query 签名对 N 个存储签名做 XOR + POPCOUNT,得 0..256 的距离。
// 每签名 4 x uint64 = 32B 连续存放,每 task 扫一段连续签名 —— DRAM 最擅长的顺序读。
//
// popcount 有两种实现,用 -DPOPCNT_SWAR 切换:
//   默认        __builtin_popcountll(RISC-V 有 Zbb 扩展时映射到 cpop 指令)
//   POPCNT_SWAR 经典 5 步掩码分治法,不依赖硬件指令
// 两者必须都通过 numpy 逐位验证 —— DRAN 项目中手写标量循环曾在本工具链 -O3 下
// 产生错误结果,不能假定整数路径必然正确。
//
// kernel 限制(SDK):堆 3 MB,栈 64 KB/task,参数 <= 9 个。

#include "mu/mu.hpp"

namespace
{
constexpr int WORDS = 4;              // 每签名 4 x uint64 = 256 位
constexpr int BITS = WORDS * 64;      // 256

#ifdef POPCNT_SWAR
// SWAR 位并行 popcount:5 步掩码分治,不依赖硬件指令。
inline int popcount64(uint64_t x)
{
    x = x - ((x >> 1) & 0x5555555555555555ull);
    x = (x & 0x3333333333333333ull) + ((x >> 2) & 0x3333333333333333ull);
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0Full;
    return static_cast<int>((x * 0x0101010101010101ull) >> 56);
}
#else
inline int popcount64(uint64_t x)
{
    return __builtin_popcountll(x);
}
#endif

// 一个签名(4 x uint64)与 query 的 Hamming 距离。
inline int hammingDist(const uint64_t* s, const uint64_t* q)
{
    int d = 0;
    for (int w = 0; w < WORDS; w++)
    {
        d += popcount64(s[w] ^ q[w]);
    }
    return d;
}

// 把 [0,numSigs) 分给 taskCount 个 task,规则是 **ceil 分块**:
// 每 task 取 ceil(n/taskCount) 个,末尾的 task 分到的更少、甚至为空
// (不是"最后一个吃掉余数" —— 两种约定给出不同划分,n=10/tc=4 时
//  ceil 分块是 [0,3)[3,6)[6,9)[9,10),余数给最后是 [0,2)[2,4)[4,6)[6,10))。
// 必须与 model_kernel.py 的 _task_range 保持同一规则,否则模型与设备静默分歧。
inline void myRange(uint64_t numSigs, uint64_t& begin, uint64_t& end)
{
    const uint64_t taskIdx = mu::getTaskIdx();
    const uint64_t taskCount = mu::getTaskCount();
    const uint64_t perTask = (numSigs + taskCount - 1) / taskCount;
    begin = taskIdx * perTask;
    end = begin + perTask;
    if (end > numSigs) end = numSigs;
    if (begin > numSigs) begin = numSigs;
}
}  // namespace

// ---------------------------------------------------------------------------
// B 组 —— 只算距离,全部写出。host 负责选 top-k。
//
//   sigs     : N x 4 uint64,连续(设备内存,已 preload)
//   query    : 4 uint64
//   numSigs  : N
//   outDists : N 个 int32,值域 [0,256]
//
// 每 task 只写自己那段,无跨 task 竞争。
// ---------------------------------------------------------------------------
void hamming_scan_dists(const uint64_t* sigs,
                        const uint64_t* query,
                        uint64_t numSigs,
                        int32_t* outDists)
{
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    uint64_t q[WORDS];
    for (int w = 0; w < WORDS; w++) q[w] = query[w];

    if (mu::getTaskIdx() == 0)
    {
        mu::hostPrintf("[kernel] taskCount=%u range=[%u,%u) q0=0x%08x%08x\n",
                       (unsigned)mu::getTaskCount(),
                       (unsigned)begin, (unsigned)end,
                       (unsigned)(q[0] >> 32), (unsigned)(q[0] & 0xFFFFFFFFu));
    }

    for (uint64_t i = begin; i < end; i++)
    {
        const int d = hammingDist(sigs + i * WORDS, q);
        outDists[i] = static_cast<int32_t>(d);
        if (mu::getTaskIdx() == 0 && i == begin)
        {
            mu::hostPrintf("[kernel] dist[%u] = %d (raw 0x%08x)\n",
                           (unsigned)i, d, (unsigned)d);
        }
    }
}

MU_KERNEL_ADD(hamming_scan_dists);
```

- [ ] **Step 2: 编译,确认无错误**

Run: `cd hamming_mu && ./build.sh`

Expected: 编译通过,`mu_kernel/mu_kernel.mubin` 更新(检查时间戳)。

- [ ] **Step 3: 看反汇编,确认 builtin 是否用上硬件指令**

Run:
```bash
source /usr/local/mu_library/mu/script/min_llvm_version_env.sh
LLVM=/usr/local/mu_library/mu_llvm/$XCENA_LLVM_VERSION/$MU_REVISION
$LLVM/bin/llvm-objdump -d hamming_mu/mu_kernel/mu_kernel.mubin \
  | grep -A40 '<hamming_scan_dists>' | grep -iE 'cpop|call|jal' | head -20
```

Expected: 若出现 `cpop` 指令 → builtin 映射到硬件 popcount(Zbb 扩展可用);若出现 `call`/`jal` 到某个 `__popcount` 库函数 → 退化为软件实现,此时 SWAR 版本很可能更快。**把观察结果记录下来**,Task 8 的对比要用。

- [ ] **Step 4: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/mu_kernel/mu_hamming.cpp
git commit -m "feat(hamming): scan kernel computing per-signature Hamming distance"
```

---

### Task 4: host 驱动 —— B 组端到端,对 numpy 逐位一致

**Files:**
- Modify: `hamming_mu/hamming_scan.cpp`(替换空骨架)

**Interfaces:**
- Consumes: Task 3 的 `hamming_scan_dists`;Task 2 的 `verify_host.py check`
- Produces: 可执行 `hamming_scan`,命令行:
  `./hamming_scan [-n N] [-k K] [-t tasks] [-b batchSize] [-s numSub] [--spread] [--mode B|C] [--dump-dists FILE] [--dump-ids FILE] [--reps R] [--seed S] [--device D]`
  本任务只实现 `--mode B`。

- [ ] **Step 1: 写 host 驱动**

把 `hamming_mu/hamming_scan.cpp` 整体替换为:

```cpp
// hamming_scan.cpp — host 驱动:MX1P 上的 Hamming 扫描与 top-k。
//
// 三组对比中的 B 组(本文件当前实现):设备算距离,回传全部 N 个距离。
// 内存:MU 核touch 的一切都必须在 Preload Region,而 preloadMemory 以
// 1 GB 对齐 + 1 GB 整数倍工作。所以只分配一个 arena,preload 一次,
// 所有 buffer 按偏移切入 —— 单独 preload 一个 32 B 的 query 会烧掉整个 1 GB 槽位。
//
// Build: 见 build.sh。

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "pxl/pxl.hpp"

namespace
{
constexpr size_t ALIGN_GB = 1ull << 30;
constexpr int WORDS = 4;                 // 每签名 4 x uint64 = 256 位
constexpr size_t SIG_BYTES = WORDS * 8;  // 32 B

inline size_t alignUpGB(size_t x) { return (x + ALIGN_GB - 1) & ~(ALIGN_GB - 1); }

double msSince(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
}

// xorshift64* —— 与 verify_host.py 无关,仅用于设备侧自生成数据;
// 需要与 numpy 比对时用 --load 指定文件。
struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 88172645463325252ull) {}
    uint64_t next()
    {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        return s;
    }
};
}  // namespace

int main(int argc, char* argv[])
{
    size_t numSigs = 1000000;
    int topK = 500;
    int taskCount = 2816;
    int numSub = 0;        // 0 = createJob() 无参数
    int batchSize = 0;     // 0 = 保持 Map 默认(16)
    int spread = 0;
    int deviceId = 0;
    int reps = 5;
    uint64_t seed = 12345;
    const char* mode = "B";
    const char* dumpDists = nullptr;
    const char* loadFile = nullptr;

    for (int i = 1; i < argc; i++)
    {
        auto next = [&](int& idx) { return std::stoll(argv[++idx]); };
        if (!strcmp(argv[i], "-n") && i + 1 < argc) numSigs = (size_t)next(i);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc) topK = (int)next(i);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) taskCount = (int)next(i);
        else if (!strcmp(argv[i], "-b") && i + 1 < argc) batchSize = (int)next(i);
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) numSub = (int)next(i);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = (int)next(i);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)next(i);
        else if (!strcmp(argv[i], "--spread")) spread = 1;
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) deviceId = (int)next(i);
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
        else if (!strcmp(argv[i], "--dump-dists") && i + 1 < argc) dumpDists = argv[++i];
        else if (!strcmp(argv[i], "--load") && i + 1 < argc) loadFile = argv[++i];
    }

    if (strcmp(mode, "B") != 0)
    {
        printf("mode %s 尚未实现(本任务只做 B)\n", mode);
        return 1;
    }

    const size_t sigBytes = numSigs * SIG_BYTES;
    printf("Hamming scan on device %d  [mode %s]\n", deviceId, mode);
    printf("  N=%zu  k=%d  tasks=%d  batchSize=%s  numSub=%s\n",
           numSigs, topK, taskCount,
           batchSize ? std::to_string(batchSize).c_str() : "default(16)",
           numSub ? std::to_string(numSub).c_str() : "default");
    printf("  signatures = %.3f GB (%zu B each)\n", sigBytes / 1e9, SIG_BYTES);

    auto context = pxl::createContext(deviceId);
    if (!context) { printf("createContext failed\n"); return 1; }

    // ---- 单 arena:MU 核 touch 的一切都在这里 ----
    size_t off = 0;
    auto reserve = [&](size_t bytes, size_t align = 4096) {
        off = (off + align - 1) & ~(align - 1);
        const size_t here = off;
        off += bytes;
        return here;
    };
    const size_t offSig   = reserve(sigBytes);
    const size_t offQuery = reserve(SIG_BYTES);
    const size_t offDists = reserve(numSigs * sizeof(int32_t));
    const size_t arenaBytes = alignUpGB(off);
    printf("  arena = %.3f GB used -> %.0f GB preloaded (%zu x 1 GB slots)\n",
           off / 1e9, arenaBytes / 1e9, arenaBytes / ALIGN_GB);

    auto* arena = reinterpret_cast<uint8_t*>(pxl::allocateMemory(deviceId, arenaBytes));
    if (!arena) { printf("allocateMemory(%.0f GB) failed\n", arenaBytes / 1e9); return 1; }

    auto* sigs  = reinterpret_cast<uint64_t*>(arena + offSig);
    auto* query = reinterpret_cast<uint64_t*>(arena + offQuery);
    auto* dists = reinterpret_cast<int32_t*>(arena + offDists);

    // ---- 填数据 ----
    auto tFill = std::chrono::steady_clock::now();
    if (loadFile)
    {
        FILE* f = fopen(loadFile, "rb");
        if (!f) { printf("打不开 %s\n", loadFile); return 1; }
        const size_t got = fread(sigs, 1, sigBytes, f);
        fclose(f);
        if (got != sigBytes)
        {
            printf("读入 %zu B,期望 %zu B —— N 与文件不匹配\n", got, sigBytes);
            return 1;
        }
        std::string qf = std::string(loadFile) + ".query";
        FILE* g = fopen(qf.c_str(), "rb");
        if (!g) { printf("打不开 %s\n", qf.c_str()); return 1; }
        if (fread(query, 1, SIG_BYTES, g) != SIG_BYTES) { printf("query 读取失败\n"); return 1; }
        fclose(g);
        printf("  loaded from %s\n", loadFile);
    }
    else
    {
        Rng rng(seed);
        for (size_t i = 0; i < numSigs * WORDS; i++) sigs[i] = rng.next();
        for (int w = 0; w < WORDS; w++) query[w] = rng.next();
    }
    printf("  fill: %.1f ms\n", msSince(tFill));

    // 把主机的写发布到设备内存。少这一步,MU 核读到的是设备 DRAM 里的旧数据,
    // 而主机自己读命中自己的缓存、看起来完全正常 —— 症状是结果错且每次不同。
    auto tFlush = std::chrono::steady_clock::now();
    pxl::flushHostCache(arena, arenaBytes);
    const double flushMs = msSince(tFlush);
    printf("  flushHostCache(%.2f GB): %.1f ms\n", arenaBytes / 1e9, flushMs);

    auto tPre = std::chrono::steady_clock::now();
    if (pxl::preloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
    {
        printf("preloadMemory failed —— 设备在 Computing 模式吗?1 GB 槽位够吗?\n");
        pxl::releaseMemory(arena);
        return 1;
    }
    const double preMs = msSince(tPre);
    printf("  preload arena: %.1f ms\n", preMs);

    // ---- 加载 MU 模块 ----
    auto module = pxl::createModule("mu_kernel/mu_kernel.mubin");
    auto job = numSub > 0 ? context->createJob(numSub) : context->createJob();
    if (job->load(module) != pxl::Result::Success) { printf("job load failed\n"); return 1; }

    auto* scanFunc = module->createFunction("hamming_scan_dists");
    auto scanExec = job->buildMap(scanFunc, taskCount);
    if (batchSize > 0) scanExec->setBatchSize(batchSize);
    if (spread) scanExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

    // ---- warmup 一轮,然后跑 reps 次取最小 ----
    if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
    { printf("warmup execute failed\n"); return 1; }
    if (scanExec->synchronize() != pxl::Result::Success)
    { printf("warmup synchronize failed\n"); return 1; }

    double best = 1e30;
    for (int r = 0; r < reps; r++)
    {
        auto t0 = std::chrono::steady_clock::now();
        if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
        { printf("execute failed\n"); return 1; }
        if (scanExec->synchronize() != pxl::Result::Success)
        { printf("synchronize failed\n"); return 1; }
        const double ms = msSince(t0);
        if (ms < best) best = ms;
        printf("  rep %d: %.3f ms\n", r, ms);
    }

    // 读结果之前必须 flush(失效方向)
    auto tRead = std::chrono::steady_clock::now();
    pxl::flushHostCache(dists, numSigs * sizeof(int32_t));
    const double readMs = msSince(tRead);

    const double scannedGB = sigBytes / 1e9;
    printf("\n--- 稳态(数据已在卡上)---\n");
    printf("  scan (min of %d)        : %8.3f ms\n", reps, best);
    printf("  有效带宽                : %8.1f GB/s   (上限 268)\n", scannedGB / (best / 1e3));
    printf("  结果回传 flush          : %8.3f ms\n", readMs);
    printf("\n--- 一次性初始化(稳态不重复)---\n");
    printf("  flushHostCache          : %8.1f ms\n", flushMs);
    printf("  preloadMemory           : %8.1f ms\n", preMs);
    printf("\n--- 链路流量 / 查询 ---\n");
    printf("  in  : query   %zu B\n", SIG_BYTES);
    printf("  out : dists   %.3f MB   <- B 组把全部距离都搬回来了\n",
           numSigs * sizeof(int32_t) / 1e6);
    printf("  设备内实扫    : %.3f GB\n", scannedGB);

    if (dumpDists)
    {
        FILE* f = fopen(dumpDists, "wb");
        if (f)
        {
            fwrite(dists, sizeof(int32_t), numSigs, f);
            fclose(f);
            printf("\n  距离已写出 -> %s\n", dumpDists);
        }
    }

    if (pxl::unloadMemory(arena, arenaBytes) != pxl::MemoryStatus::Success)
        printf("warning: unloadMemory failed\n");
    pxl::releaseMemory(arena);
    context->destroyJob(job);
    pxl::destroyContext(context);
    return 0;
}
```

- [ ] **Step 2: 编译**

Run: `cd hamming_mu && ./build.sh`

Expected: 编译通过。

- [ ] **Step 3: 小规模跑,与 numpy 逐位比对**

Run:
```bash
python hamming_mu/verify_host.py gen -n 10000 -o /tmp/sig10k.bin
cd hamming_mu
./hamming_scan -n 10000 -t 64 --load /tmp/sig10k.bin --dump-dists /tmp/dev10k.i32
cd -
python hamming_mu/verify_host.py check --ref /tmp/sig10k.bin.dists.npy --dists /tmp/dev10k.i32
```

Expected: `PASS: 10000 个距离逐位一致`

**若 FAIL,按此顺序诊断(不要跳过顺序):**
1. **结果错 + 每次运行结果不同 + 主机自查正常** → 三条同时出现就是 `flushHostCache` 问题,先查 flush 调用位置,**不要**去怀疑算法或编译器。
2. 只错部分且稳定复现 → 看 kernel 的 `hostPrintf` 原始比特输出,与 host 侧同一个值的 hex 对比。浮点/十进制格式化会掩盖差异,原始比特不会。
3. 改了代码但行为没变 → 加一行新的 `hostPrintf` 看它有没有出现,确认真的重新编译并重传了。

- [ ] **Step 4: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/hamming_scan.cpp
git commit -m "feat(hamming): host driver for mode B, bit-exact vs numpy"
```

---

### Task 5: 扫 batchSize 与 numSub,建立真实基线

**Files:**
- Create: `hamming_mu/sweep.sh`
- Create: `hamming_mu/RESULTS.md`

**Interfaces:**
- Consumes: Task 4 的 `hamming_scan --mode B`
- Produces: `hamming_mu/RESULTS.md` 中记录的最优 `batchSize` / `numSub`,后续所有测量都用这组参数

**为什么必须现在做:** `setBatchSize` 默认 16,而 `活跃核数 ≤ ceil(taskCount/batchSize)`。
`buildMap(fn, 2816)` + 默认 batchSize 只激活 `2816/16 = 176` 个核,其余 2640 个闲置 ——
**16 倍差距**。晚扫会让后续所有测量建立在错误基线上。

- [ ] **Step 1: 写扫描脚本**

创建 `hamming_mu/sweep.sh`:

```bash
#!/bin/bash
# 扫 batchSize 与 numSub。必须在任何性能结论之前跑。
# 用法: ./sweep.sh [N] [tasks]
set -u
N="${1:-10000000}"
T="${2:-2816}"

echo "=== batchSize sweep (N=$N tasks=$T) ==="
printf "%-12s %-12s\n" "batchSize" "scan_ms"
for b in 1 2 3 4 8 16; do
    ms=$(./hamming_scan -n "$N" -t "$T" -b "$b" --reps 5 2>/dev/null \
         | grep 'scan (min' | awk '{print $6}')   # $5 是冒号,$6 才是数字
    printf "%-12s %-12s\n" "$b" "${ms:-FAIL}"
done

echo
echo "=== numSub sweep (batchSize 用上面的最优值,手工填入 BEST_B) ==="
BEST_B="${BEST_B:-4}"
printf "%-12s %-12s\n" "numSub" "scan_ms"
for s in 4 8 16 22; do
    ms=$(./hamming_scan -n "$N" -t "$T" -b "$BEST_B" -s "$s" --reps 5 2>/dev/null \
         | grep 'scan (min' | awk '{print $6}')   # $5 是冒号,$6 才是数字
    printf "%-12s %-12s\n" "$s" "${ms:-FAIL}"
done
```

- [ ] **Step 2: 跑 batchSize 扫描**

Run:
```bash
cd hamming_mu && chmod +x sweep.sh && ./sweep.sh 10000000 2816
```

Expected: 6 行 batchSize 结果。预期形状是**中间有谷**:b=1 因分派开销放大而慢,b=16 因并行度不足而慢,最优通常在 **b=3~4**。若最优就是 16,说明本负载的分派开销特征与 DRAN 不同 —— 记录下来,不要强行套用 DRAN 的结论。

- [ ] **Step 3: 用最优 batchSize 跑 numSub 扫描**

Run:
```bash
cd hamming_mu && BEST_B=<上一步的最优值> ./sweep.sh 10000000 2816
```

Expected: 4 行 numSub 结果。

- [ ] **Step 4: 记录结果**

创建 `hamming_mu/RESULTS.md`:

```markdown
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
```

把实测数字填进去。

- [ ] **Step 5: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/sweep.sh hamming_mu/RESULTS.md
git commit -m "perf(hamming): sweep batchSize/numSub, establish real baseline"
```

---

### Task 6: C 组 kernel —— 计数排序 top-k 与溢出处理

**Files:**
- Modify: `hamming_mu/mu_kernel/mu_hamming.cpp`(追加两个 kernel)

**Interfaces:**
- Consumes: Task 3 的 `popcount64` / `hammingDist` / `myRange`
- Produces: 两个 kernel 符号:
  ```cpp
  void hamming_scan_hist(const uint64_t* sigs, const uint64_t* query,
                         uint64_t numSigs, int32_t coarseThresh,
                         int32_t candCap, uint32_t* outHist,
                         uint32_t* outCand, uint32_t* outMeta)
  // outHist : taskCount * 257 uint32   每 task 的距离直方图
  // outCand : taskCount * candCap * 2 uint32  每 task 的候选分片,
  //           交错存放 (id, dist):cand[2j]=id, cand[2j+1]=dist。
  //           ★ 必须连距离一起存 —— merge 要按 exactThresh 过滤。
  //           candCap 计的是**条目数**,故缓冲字节数 = taskCount*candCap*2*4。
  // outMeta : taskCount * 2 uint32     [0]=candCount  [1]=overflow(0/1)

  void hamming_merge_topk(const uint32_t* hist, const uint32_t* cand,
                          const uint32_t* meta, int32_t taskCount,
                          int32_t candCap, int32_t topK,
                          const int32_t* dists, uint32_t* outIds)
  // outIds : (topK + 3) uint32 —— 前 topK 个是 id(不足处填 0xFFFFFFFF),
  //          尾部 3 个槽位是给 host 的状态回传:
  //            outIds[topK]   = exactThresh(第 k 名的精确距离阈值)
  //            outIds[topK+1] = anyOverflow(0/1)
  //            outIds[topK+2] = collected(实际收集到的 id 个数)
  // dists  : 当前未使用(保留参数位);重跑由 host 驱动收紧阈值后重新执行
  ```

  **host 必须为 outIds 分配 `topK + 3` 个 uint32。**

**★ 候选必须带距离(一个已复现的静默错误):**
初版设计里候选只存 id、merge 按索引序取前 k 个 —— **错的,而且是静默的**。
候选只满足 `d ≤ coarseThresh`;当 `coarseThresh > exactThresh` 时,按索引序
取前 k 个会取到宽带里的任意 id。实测(n=2000, k=50, coarse=148, cap 足够大):
`overflow=0, collected=50 → 成功门通过`,但返回 id 的最大距离 **145**,
而真实第 k 名距离是 **113**。成功门通过、结果却是错的。

**修法:候选交错存 (id, dist),merge 按 `exactThresh` 真的过滤。**
由此得到关键性质:**过滤后每个返回 id 都满足 `d ≤ exactThresh = 第k名距离`,
所以溢出不再可能产生错误结果,只可能导致收集不足** ——
失败模式从"静默返回错的"降级为"响亮地收集不够"。

**溢出为什么能自我修复:** 直方图统计**全部**签名,不受候选缓冲容量影响,
所以 `exactThresh` **始终可算**。主机据此调整:收集不足就令
`coarseThresh = exactThresh` 重跑(过紧则放宽、过松则收紧,两个方向都收敛);
在 `exactThresh` 仍不足则说明 `candCap` 太小。

**算法已在 Task 2b 用 Python 建模并模糊测试通过** —— 
`hamming_mu/model_kernel.py` 是本 kernel 的可执行参考,逐行对应。
写 C++ 时如与模型不一致,以模型为准(它验证过,C++ 无法本地编译)。

- [ ] **Step 1: 追加两个 kernel**

在 `hamming_mu/mu_kernel/mu_hamming.cpp` 的 `MU_KERNEL_ADD(hamming_scan_dists);` **之前**,插入:

```cpp
// ---------------------------------------------------------------------------
// C 组 stage 1 —— 扫描 + 每 task 257 桶直方图 + 候选收集。
//
//   coarseThresh : 粗阈值,dist <= 此值的候选收进分片
//   candCap      : 每 task 候选**条目数**上限(每条目 2 个 uint32)
//   outHist      : taskCount * 257 uint32
//   outCand      : taskCount * candCap * 2 uint32 —— 交错 (id, dist)
//   outMeta      : taskCount * 2 uint32 —— [0]=candCount [1]=overflow
//
// ★ 候选必须连距离一起存。只存 id 的话 merge 无法按 exactThresh 过滤,
//   粗阈值偏大时会静默返回宽带里的任意 id(实测:返回距离 145,真实第k名 113,
//   而成功门仍然通过)。详见 Task 2b 的模型与回归用例。
//
// 直方图放在堆上而非栈上:257 * 4 = 1028 B 虽然放得下 64 KB 栈,但显式用
// 传入的设备内存分片更省栈,也便于 merge kernel 直接读。
// ---------------------------------------------------------------------------
void hamming_scan_hist(const uint64_t* sigs,
                       const uint64_t* query,
                       uint64_t numSigs,
                       int32_t coarseThresh,
                       int32_t candCap,
                       uint32_t* outHist,
                       uint32_t* outCand,
                       uint32_t* outMeta)
{
    const uint32_t taskIdx = mu::getTaskIdx();
    uint64_t begin, end;
    myRange(numSigs, begin, end);

    uint64_t q[WORDS];
    for (int w = 0; w < WORDS; w++) q[w] = query[w];

    uint32_t* myHist = outHist + (uint64_t)taskIdx * (BITS + 1);
    uint32_t* myCand = outCand + (uint64_t)taskIdx * (uint64_t)candCap * 2;  // 每条目 2 个 uint32
    uint32_t* myMeta = outMeta + (uint64_t)taskIdx * 2;

    for (int b = 0; b <= BITS; b++) myHist[b] = 0u;

    uint32_t nCand = 0;
    uint32_t overflow = 0;

    for (uint64_t i = begin; i < end; i++)
    {
        const int d = hammingDist(sigs + i * WORDS, q);
        myHist[d]++;                       // 距离有界 -> 计数排序的基础
        if (d <= coarseThresh)
        {
            if (nCand < (uint32_t)candCap)
            {
                myCand[2 * nCand] = (uint32_t)i;      // id
                myCand[2 * nCand + 1] = (uint32_t)d;  // ★ 距离必须一起存
                nCand++;
            }
            else overflow = 1u;            // 记录但不中断:直方图仍然完整
        }
    }

    myMeta[0] = nCand;
    myMeta[1] = overflow;

    if (taskIdx == 0)
    {
        mu::hostPrintf("[kernel] task0 range=[%u,%u) nCand=%u overflow=%u\n",
                       (unsigned)begin, (unsigned)end, (unsigned)nCand, (unsigned)overflow);
    }
}

// ---------------------------------------------------------------------------
// C 组 stage 2 —— 单 task 归并。
//
// 1. 归并所有 task 的直方图
// 2. 从距离 0 累加,定位第 topK 名所在的桶 -> 精确阈值 exactThresh
// 3. 扫候选分片收集 id;若任一 task 溢出且 dists 非空,则退回扫 dists 补齐
//
// 必须单独一个 kernel:归并要看到所有 task 的直方图,而 Map 内无跨 task 同步。
// ---------------------------------------------------------------------------
void hamming_merge_topk(const uint32_t* hist,
                        const uint32_t* cand,
                        const uint32_t* meta,
                        int32_t taskCount,
                        int32_t candCap,
                        int32_t topK,
                        const int32_t* dists,
                        uint32_t* outIds)
{
    if (mu::getTaskIdx() != 0) return;

    // --- 1. 归并直方图 ---
    uint32_t total[BITS + 1];
    for (int b = 0; b <= BITS; b++) total[b] = 0u;
    for (int t = 0; t < taskCount; t++)
    {
        const uint32_t* h = hist + (uint64_t)t * (BITS + 1);
        for (int b = 0; b <= BITS; b++) total[b] += h[b];
    }

    // --- 2. 定位第 topK 名的精确距离阈值 ---
    int exactThresh = BITS;
    uint64_t cum = 0;
    for (int b = 0; b <= BITS; b++)
    {
        cum += total[b];
        if (cum >= (uint64_t)topK) { exactThresh = b; break; }
    }

    // --- 3. 检查是否有 task 溢出 ---
    int anyOverflow = 0;
    for (int t = 0; t < taskCount; t++)
    {
        if (meta[(uint64_t)t * 2 + 1]) { anyOverflow = 1; break; }
    }

    mu::hostPrintf("[kernel] exactThresh=%d anyOverflow=%d\n", exactThresh, anyOverflow);

    // ★ 只收 dist <= exactThresh 的候选。没有这个过滤,粗阈值偏大时会返回
    //   宽带里的任意 id —— 已复现的静默错误(返回距离 145,真实第k名 113)。
    //   有了它,每个返回 id 必然满足 d <= 第k名距离,溢出便不再能产生错误结果。
    int n = 0;
    for (int t = 0; t < taskCount && n < topK; t++)
    {
        const uint32_t cnt = meta[(uint64_t)t * 2];
        const uint32_t* c = cand + (uint64_t)t * (uint64_t)candCap * 2;
        for (uint32_t j = 0; j < cnt && n < topK; j++)
        {
            if ((int)c[2 * j + 1] <= exactThresh) outIds[n++] = c[2 * j];
        }
    }

    // 不足 topK 时用哨兵填充,host 可据此判断收集是否完整
    const int collected = n;
    for (; n < topK; n++) outIds[n] = 0xFFFFFFFFu;

    // outIds[topK] = exactThresh, outIds[topK+1] = anyOverflow, outIds[topK+2] = collected
    // (host 端为 outIds 多分配了 3 个槽位)
    outIds[topK]     = (uint32_t)exactThresh;
    outIds[topK + 1] = (uint32_t)anyOverflow;
    outIds[topK + 2] = (uint32_t)collected;

    (void)dists;  // 保留参数位以便将来做设备内重扫;当前重跑由 host 驱动
}
```

并在文件末尾追加注册:

```cpp
MU_KERNEL_ADD(hamming_scan_hist);
MU_KERNEL_ADD(hamming_merge_topk);
```

- [ ] **Step 2: 检查栈用量**

`hamming_merge_topk` 里 `uint32_t total[BITS + 1]` = 257 × 4 = **1028 B**,
远低于 64 KB 栈限制,安全。`hamming_scan_hist` 无大局部数组(直方图写在设备内存)。

Run: `cd hamming_mu && ./build.sh`

Expected: 编译通过,无栈相关警告。

- [ ] **Step 3: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/mu_kernel/mu_hamming.cpp
git commit -m "feat(hamming): counting-sort top-k kernels with overflow detection"
```

---

### Task 7: host 驱动 C 组 —— 端到端 top-k 与溢出重跑

**Files:**
- Modify: `hamming_mu/hamming_scan.cpp`

**Interfaces:**
- Consumes: Task 6 的 `hamming_scan_hist` / `hamming_merge_topk`
- Produces: `--mode C` 可用;`--dump-ids FILE` 写出 topK 个 uint32

- [ ] **Step 1: 加 C 模式的 buffer 与执行路径**

在 `hamming_scan.cpp` 中做三处修改。

**(a) 放开 mode 检查** —— 把:

```cpp
    if (strcmp(mode, "B") != 0)
    {
        printf("mode %s 尚未实现(本任务只做 B)\n", mode);
        return 1;
    }
```

替换为:

```cpp
    const bool modeC = (strcmp(mode, "C") == 0);
    if (!modeC && strcmp(mode, "B") != 0)
    {
        printf("mode 必须是 B 或 C\n");
        return 1;
    }
    // 粗阈值:随机 256 位签名的距离集中在 128 附近,取 118 可覆盖 topK
    // 而不至于收进过多候选。溢出时会自动重跑收紧。
    int coarseThresh = 118;
    int candCap = topK * 4;   // 每 task 候选容量
```

**(b) 增加 C 模式的 arena 分片** —— 在 `const size_t offDists = ...` 之后插入:

```cpp
    const size_t offHist = modeC ? reserve((size_t)taskCount * 257 * sizeof(uint32_t)) : 0;
    // 候选交错存 (id, dist) -> 每条目 2 个 uint32
    const size_t offCand = modeC ? reserve((size_t)taskCount * candCap * 2 * sizeof(uint32_t)) : 0;
    const size_t offMeta = modeC ? reserve((size_t)taskCount * 2 * sizeof(uint32_t)) : 0;
    // outIds 需要 topK + 3 个槽位:尾部 3 个是 kernel 回传的
    // exactThresh / anyOverflow / collected
    const size_t offIds  = modeC ? reserve((size_t)(topK + 3) * sizeof(uint32_t)) : 0;
```

并在 `auto* dists = ...` 之后插入:

```cpp
    auto* hist = modeC ? reinterpret_cast<uint32_t*>(arena + offHist) : nullptr;
    auto* cand = modeC ? reinterpret_cast<uint32_t*>(arena + offCand) : nullptr;
    auto* meta = modeC ? reinterpret_cast<uint32_t*>(arena + offMeta) : nullptr;
    auto* ids  = modeC ? reinterpret_cast<uint32_t*>(arena + offIds)  : nullptr;
```

**(c) 增加 C 模式的执行块** —— 把从 `auto* scanFunc = module->createFunction("hamming_scan_dists");`
到 `--dump-dists` 写文件那段之间的执行与计时代码,替换为:

```cpp
    double best = 1e30, bestScan = 1e30, bestMerge = 1e30;
    size_t outBytes = 0;

    if (!modeC)
    {
        auto* scanFunc = module->createFunction("hamming_scan_dists");
        auto scanExec = job->buildMap(scanFunc, taskCount);
        if (batchSize > 0) scanExec->setBatchSize(batchSize);
        if (spread) scanExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

        // warmup —— 返回值必须检查。一次静默失败的 warmup 会让后面所有计时
        // 建立在"kernel 其实没跑起来"的基础上,而计时循环自己是检查了的,
        // 于是错误会以"数字异常好看"的形式出现,最难察觉。
        if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
        { printf("warmup execute failed\n"); return 1; }
        if (scanExec->synchronize() != pxl::Result::Success)
        { printf("warmup synchronize failed\n"); return 1; }

        for (int r = 0; r < reps; r++)
        {
            auto t0 = std::chrono::steady_clock::now();
            if (scanExec->execute(sigs, query, (uint64_t)numSigs, dists) != pxl::Result::Success)
            { printf("execute failed\n"); return 1; }
            if (scanExec->synchronize() != pxl::Result::Success)
            { printf("synchronize failed\n"); return 1; }
            const double ms = msSince(t0);
            if (ms < best) best = ms;
            printf("  rep %d: %.3f ms\n", r, ms);
        }
        bestScan = best;
        outBytes = numSigs * sizeof(int32_t);
        pxl::flushHostCache(dists, outBytes);
    }
    else
    {
        auto* histFunc  = module->createFunction("hamming_scan_hist");
        auto* mergeFunc = module->createFunction("hamming_merge_topk");
        auto histExec  = job->buildMap(histFunc, taskCount);
        auto mergeExec = job->buildMap(mergeFunc, 1);   // 单 task:归约
        if (batchSize > 0) histExec->setBatchSize(batchSize);
        if (spread) histExec->setLocalityMode(pxl::LocalityMode::SpreadMode);

        auto runOnce = [&](double& scanMs, double& mergeMs) -> bool {
            auto t0 = std::chrono::steady_clock::now();
            if (histExec->execute(sigs, query, (uint64_t)numSigs, (int32_t)coarseThresh,
                                  (int32_t)candCap, hist, cand, meta) != pxl::Result::Success)
                return false;
            if (histExec->synchronize() != pxl::Result::Success) return false;
            scanMs = msSince(t0);

            auto t1 = std::chrono::steady_clock::now();
            if (mergeExec->execute(hist, cand, meta, (int32_t)taskCount, (int32_t)candCap,
                                   (int32_t)topK, dists, ids) != pxl::Result::Success)
                return false;
            if (mergeExec->synchronize() != pxl::Result::Success) return false;
            mergeMs = msSince(t1);
            return true;
        };

        double s = 0, m = 0;
        if (!runOnce(s, m)) { printf("warmup failed\n"); return 1; }   // warmup

        // 重试循环。merge 已按 exactThresh 过滤候选,所以收集到的 id 必然满足
        // d <= 第k名距离 —— **溢出不会产生错误结果**,只可能导致收集不足。
        // 因此重试条件只看数量,overflow 退化为诊断信息。
        // kernel 把状态写在 outIds 尾部三个槽位,读之前必须 flush(失效方向)。
        for (int attempt = 0; attempt < 5; attempt++)
        {
            pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
            const int      exactThresh = (int)ids[topK];
            const uint32_t overflow    = ids[topK + 1];
            const uint32_t collected   = ids[topK + 2];

            if (collected >= (uint32_t)topK)
            {
                if (overflow)
                    printf("  注:发生过候选溢出,但结果仍有效(全部 d <= 精确阈值 %d)\n",
                           exactThresh);
                break;
            }
            if (coarseThresh == exactThresh)
            {
                printf("  已在精确阈值 %d 仍只收集到 %u/%d -> candCap(%d)太小\n",
                       exactThresh, collected, topK, candCap);
                break;                       // 重试无用,需增大 candCap
            }
            // 过紧则放宽、过松则收紧 —— 两个方向都收敛到精确阈值
            printf("  收集不足(%u/%d)-> coarseThresh %d -> 精确阈值 %d 重跑\n",
                   collected, topK, coarseThresh, exactThresh);
            coarseThresh = exactThresh;
            if (!runOnce(s, m)) { printf("rescan failed\n"); return 1; }
        }
        pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
        if (ids[topK + 2] < (uint32_t)topK)
        {
            printf("警告:仅收集到 %u/%d 个 id(candCap=%d 太小,提高它重试)\n",
                   ids[topK + 2], topK, candCap);
        }

        for (int r = 0; r < reps; r++)
        {
            if (!runOnce(s, m)) { printf("execute failed\n"); return 1; }
            if (s < bestScan) bestScan = s;
            if (m < bestMerge) bestMerge = m;
            printf("  rep %d: scan %.3f ms  merge %.3f ms\n", r, s, m);
        }
        best = bestScan + bestMerge;
        // 链路流量只算真正的结果(k 个 id);尾部 3 个状态槽是调试用,
        // 生产实现里不需要,故不计入对外报告的流量。
        outBytes = (size_t)topK * sizeof(uint32_t);
        pxl::flushHostCache(ids, (size_t)(topK + 3) * sizeof(uint32_t));
    }
```

**(d) 更新报告与写出** —— 把原来的报告块与 `--dump-dists` 块替换为:

```cpp
    const double scannedGB = sigBytes / 1e9;
    printf("\n--- 稳态(数据已在卡上)---\n");
    printf("  scan (min of %d)        : %8.3f ms\n", reps, bestScan);
    if (modeC) printf("  merge/top-k             : %8.3f ms\n", bestMerge);
    printf("  合计每查询              : %8.3f ms\n", best);
    printf("  扫描有效带宽            : %8.1f GB/s   (上限 268)\n",
           scannedGB / (bestScan / 1e3));
    printf("\n--- 一次性初始化(稳态不重复)---\n");
    printf("  flushHostCache          : %8.1f ms\n", flushMs);
    printf("  preloadMemory           : %8.1f ms\n", preMs);
    printf("\n--- 链路流量 / 查询 ---\n");
    printf("  in  : query   %zu B\n", SIG_BYTES);
    printf("  out : %s  %.6f MB\n", modeC ? "top-k ids" : "全部距离", outBytes / 1e6);
    printf("  设备内实扫    : %.3f GB\n", scannedGB);
    printf("  流量压缩比    : %.0fx\n", (double)sigBytes / (double)(outBytes + SIG_BYTES));

    if (dumpDists && !modeC)
    {
        FILE* f = fopen(dumpDists, "wb");
        if (f) { fwrite(dists, sizeof(int32_t), numSigs, f); fclose(f);
                 printf("\n  距离已写出 -> %s\n", dumpDists); }
    }
    if (dumpIds && modeC)
    {
        FILE* f = fopen(dumpIds, "wb");
        if (f) { fwrite(ids, sizeof(uint32_t), topK, f); fclose(f);
                 printf("\n  top-k ids 已写出 -> %s\n", dumpIds); }
    }
```

并在参数解析处补上 `--dump-ids`:在 `else if (!strcmp(argv[i], "--dump-dists") ...` 之后加一行:

```cpp
        else if (!strcmp(argv[i], "--dump-ids") && i + 1 < argc) dumpIds = argv[++i];
```

以及在变量声明处 `const char* dumpDists = nullptr;` 之后加:

```cpp
    const char* dumpIds = nullptr;
```

- [ ] **Step 2: 编译**

Run: `cd hamming_mu && ./build.sh`

Expected: 编译通过。

- [ ] **Step 3: 小规模验证 top-k 正确**

Run:
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 64 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/dev10k.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/dev10k.ids -k 100
```

Expected: `PASS: 100 个 id 距离全部 <= 第k名(...)`

**注意平局语义**:随机签名距离集中,第 k 名附近常有大量平局。判据是
"返回的每个 id 的距离必须 ≤ 第 k 名的距离",而**不是** id 集合完全相同 ——
计数排序与 numpy argsort 在平局时选的具体 id 可以不同,这不是错误。

- [ ] **Step 4: 验证溢出路径真的会触发并自愈**

Run:
```bash
cd hamming_mu
./hamming_scan -n 10000 -k 100 -t 4 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/ovf.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/ovf.ids -k 100
```

(只用 4 个 task,每 task 要扫 2500 个签名,候选远超 `candCap = 400`,必然溢出。)

Expected: kernel 端打印 `[kernel] task0 ... overflow=1`,且最终 `check-topk` **PASS**。

主机侧会出现下面两种之一,都是正确行为:
- `注:发生过候选溢出,但结果仍有效(全部 d <= 精确阈值 N)` ——
  溢出了但仍收集够 k 个。因为 merge 已按 exactThresh 过滤,这些 id **全部合格**,
  无需重跑。这正是 Option B 带来的性质。
- `收集不足(x/100)-> coarseThresh 118 -> 精确阈值 N 重跑` ——
  溢出丢掉了太多合格项,收敛到精确阈值后重跑。

**关键判据是 `check-topk` PASS**,它现在同时检查距离与 id 唯一性(Task 2 已加固)。
若看到 `已在精确阈值 N 仍只收集到 x/100 -> candCap 太小`,把 `candCap` 调大重试。

- [ ] **Step 5: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/hamming_scan.cpp
git commit -m "feat(hamming): mode C end-to-end top-k with overflow retry"
```

---

### Task 8: A 组 host 基线 + 三组对比出表

**Files:**
- Create: `hamming_mu/bench_host_baseline.py`
- Modify: `hamming_mu/RESULTS.md`

**Interfaces:**
- Consumes: Task 5 的最优 batchSize/numSub;Task 4/7 的 B/C 两模式
- Produces: `RESULTS.md` 中的三组对比表(论文用)

- [ ] **Step 1: 写 host 基线**

创建 `hamming_mu/bench_host_baseline.py`:

```python
"""A 组基线:host 侧 Hamming 扫描 + top-k。

必须与设备测量在同一台机器上跑 —— 跨机器数字不可比。
用位打包 uint64 + np.bitwise_count(numpy >= 2.0);无该函数时退回查表法。
"""
import argparse
import time

import numpy as np

WORDS = 4


def scan_topk(sigs, query, k):
    x = np.bitwise_xor(sigs, query)
    if hasattr(np, "bitwise_count"):
        d = np.bitwise_count(x).sum(axis=1)
    else:
        d = np.unpackbits(x.view(np.uint8).reshape(-1, WORDS * 8), axis=1).sum(axis=1)
    idx = np.argpartition(d, k)[:k]
    return idx[np.argsort(d[idx], kind="stable")]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=10_000_000)
    p.add_argument("-k", type=int, default=500)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    sigs = rng.integers(0, 2**64, size=(a.n, WORDS), dtype=np.uint64)
    query = rng.integers(0, 2**64, size=WORDS, dtype=np.uint64)

    scan_topk(sigs, query, a.k)   # warmup

    best = float("inf")
    for r in range(a.reps):
        t0 = time.perf_counter()
        scan_topk(sigs, query, a.k)
        ms = (time.perf_counter() - t0) * 1e3
        best = min(best, ms)
        print(f"  rep {r}: {ms:.3f} ms")

    gb = a.n * 32 / 1e9
    print(f"\nA 组 host 基线  N={a.n}  k={a.k}")
    print(f"  最小耗时      : {best:.3f} ms")
    print(f"  有效带宽      : {gb / (best / 1e3):.1f} GB/s")
    print(f"  需入主机的数据: {gb:.3f} GB")
    print(f"  numpy         : {np.__version__}  bitwise_count={hasattr(np, 'bitwise_count')}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑 A 组基线**

Run: `python hamming_mu/bench_host_baseline.py -n 10000000 -k 500 --reps 5`

Expected: 输出最小耗时与有效带宽。记录 numpy 版本与是否用上 `bitwise_count`。

- [ ] **Step 3: 跑 B、C 两组(用 Task 5 的最优参数)**

Run:
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

Expected: 6 组输出。N=100M 时 arena 约 3.2 GB(mode B 另需 400 MB 距离数组),
需确认 Preload Region 有足够 1 GB 槽位;若 `preloadMemory failed`,降到 N=50M 并在
`RESULTS.md` 中注明规模上限。

- [ ] **Step 4: popcount 双实现对比**

Run:
```bash
cd hamming_mu
# 默认 builtin 版已在上面测过,现在编 SWAR 版
sed -i 's/^set(compile_options/set(compile_options\n    -DPOPCNT_SWAR/' mu_kernel/CMakeLists.txt
./build.sh
./hamming_scan -n 10000000 -k 500 -t 2816 -b $B -s $S --mode C --reps 5 | grep -E 'scan \(min|有效带宽'
# 验证 SWAR 版同样正确
./hamming_scan -n 10000 -k 100 -t 64 --mode C --load /tmp/sig10k.bin --dump-ids /tmp/swar.ids
cd -
python hamming_mu/verify_host.py check-topk --ref /tmp/sig10k.bin.dists.npy --ids /tmp/swar.ids -k 100
```

Expected: SWAR 版 `check-topk` 必须同样 `PASS`(DRAN 的教训:不能假定手写整数路径
在 -O3 下必然正确)。记录两版的耗时差异。

测完后恢复 builtin 版(若 builtin 更快):
```bash
cd hamming_mu && sed -i '/-DPOPCNT_SWAR/d' mu_kernel/CMakeLists.txt && ./build.sh && cd -
```

- [ ] **Step 5: 填写对比表**

在 `hamming_mu/RESULTS.md` 末尾追加,并填入实测数字:

```markdown
## 三组对比(k=500,参数 -t 2816 -b <最优> -s <最优>)

| N | 方法 | 每查询延迟 (ms) | 链路流量/查询 | 设备内实扫 |
|---|---|---|---|---|
| 1M | A host-only | | 32 MB 需入主机 | — |
| 1M | B 设备扫+host选 | | 4.0 MB | 32 MB |
| 1M | C 全下沉 | | 2.0 KB | 32 MB |
| 10M | A host-only | | 320 MB 需入主机 | — |
| 10M | B 设备扫+host选 | | 40 MB | 320 MB |
| 10M | C 全下沉 | | 2.0 KB | 320 MB |
| 100M | A host-only | | 3.2 GB 需入主机 | — |
| 100M | B 设备扫+host选 | | 400 MB | 3.2 GB |
| 100M | C 全下沉 | | 2.0 KB | 3.2 GB |

**读法:**
- **A ↔ C** = 近内存的价值
- **B ↔ C** = selection 下沉的价值

## popcount 实现对比 (N=10M, mode C)

| 实现 | scan (ms) | 有效带宽 (GB/s) | 正确性 |
|---|---|---|---|
| `__builtin_popcountll` | | | PASS |
| SWAR 5 步掩码 | | | PASS |

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
```

- [ ] **Step 6: 提交**

```bash
cd "$HOME/engramme"
git add hamming_mu/bench_host_baseline.py hamming_mu/RESULTS.md
git commit -m "bench(hamming): host baseline + three-way comparison results"
```

---

### Task 9: 汇总服务器 RUNBOOK [本地]

**Files:**
- Create/Modify: `hamming_mu/RUNBOOK.md`
- Create: `hamming_mu/transfer.sh`

**Interfaces:**
- Consumes: Task 1~8 中所有标记 [服务器] 的步骤
- Produces: 一份可从上到下照着执行的服务器操作手册

**背景:** 开发机无 XCENA 硬件也无 C++ 编译器,所有编译与测量都要在
另一台服务器上进行。本任务把散落在各任务中的服务器步骤汇总成一份连贯手册,
使执行者不必回头翻计划。

- [ ] **Step 1: 写传输脚本 [本地]**

创建 `hamming_mu/transfer.sh`:

```bash
#!/bin/bash
# 把源码同步到有 XCENA 的服务器。只传源码,不传数据与构建产物。
# 用法: ./transfer.sh user@server:/path/to/dest
set -eu
DEST="${1:?用法: ./transfer.sh user@server:/path/to/dest}"

rsync -av --progress \
  --exclude 'build/' \
  --exclude '*.mubin' \
  --exclude '*.bin' --exclude '*.i32' --exclude '*.ids' --exclude '*.npy' \
  --exclude '__pycache__/' \
  ./ "${DEST}/hamming_mu/"

echo
echo "同步完成。在服务器上执行:"
echo "  cd ${DEST##*:}/hamming_mu && bash env_check.sh"
```

- [ ] **Step 2: 汇总 RUNBOOK [本地]**

创建(或补全)`hamming_mu/RUNBOOK.md`,内容为按顺序排列的服务器操作,
每节须包含:**命令、预期输出、失败时怎么办**。骨架如下,
把各任务中标记 [服务器] 的步骤逐条填入:

```markdown
# 服务器操作手册 —— MX1P Hamming 扫描

开发机无 XCENA 硬件与 C++ 工具链,以下全部在装有 MX1P 的服务器上执行。
算法已在开发机用 `model_kernel.py` 模糊测试通过,故此处出的问题
优先怀疑**环境**与**性能**,而非算法。

## 0. 同步源码

    ./transfer.sh user@server:/home/you/work

## 1. 环境校验(不过不要继续)

    cd hamming_mu && bash env_check.sh

预期:PCI 设备存在;dkms 有匹配当前内核的 mx_dma;daxctl 显示 devdax;
computable=Yes;validate_host.sh 无 FAIL。

失败处理:
- mode 是 system-ram → `sudo daxctl reconfigure-device --mode=devdax --force dax0.0`
- dkms 无条目 → 换过内核,需重建 mx_dma
- computable=No → 固件不在 Computing 模式

## 2. 编译

    ./build.sh

预期:产出 `mu_kernel/mu_kernel.mubin` 与可执行 `hamming_scan`。
失败处理:`half_float 未声明` → 本项目为纯整数路径,不应 include half.hpp。

## 3. 看反汇编,确认 popcount 是否用上硬件指令

(填入 Task 3 Step 3 的命令与判读方法)

## 4. 正确性验证(性能之前必须过)

(填入 Task 4 Step 3 的 B 组逐位比对、Task 7 Step 3/4 的 C 组 top-k 与溢出验证)

**若结果错 + 每次运行不同 + 主机自查正常 → 三条同时出现就是 flushHostCache 问题,
先查 flush,不要怀疑算法。**

## 5. 扫 batchSize / numSub(建立真实基线,不可后移)

(填入 Task 5 的命令;强调默认 batchSize=16 只激活 1/16 的核)

## 6. 三组对比测量

(填入 Task 8 Step 2/3 的 A/B/C 三组命令)

## 7. popcount 双实现对比

(填入 Task 8 Step 4 的命令,含 SWAR 版必须重新验证正确性)

## 8. 把结果填回 RESULTS.md 并传回开发机
```

- [ ] **Step 3: 自查 RUNBOOK 完整性 [本地]**

逐条核对:Task 1 Step 2/6、Task 3 Step 2/3、Task 4 Step 2/3、Task 5 全部、
Task 7 Step 2/3/4、Task 8 Step 2/3/4 —— 这些 [服务器] 步骤是否都已进入 RUNBOOK。

Run: `grep -c '^##' hamming_mu/RUNBOOK.md`

Expected: ≥ 9 个小节(0~8)。

- [ ] **Step 4: 提交 [本地]**

```bash
cd "$HOME/engramme"
git add hamming_mu/RUNBOOK.md hamming_mu/transfer.sh
git commit -m "docs(hamming): server runbook and transfer script"
```

---

## Self-Review 记录

**Spec 覆盖检查:**

| Spec 章节 | 对应任务 |
|---|---|
| §2 三组对比 A/B/C | Task 4(B)、Task 7(C)、Task 8(A + 出表) |
| §3 数据与布局(合成、4×uint64 连续、单 arena) | Task 2(生成)、Task 4(arena 切分) |
| §4 kernel 1 直方图 + 候选 | Task 6 |
| §4 kernel 2 归并定阈值 | Task 6 |
| §4 候选溢出处理 | Task 6(kernel 端检测)、Task 7 Step 4(端到端验证自愈) |
| §5 popcount 双实现 | Task 3(实现 + 反汇编)、Task 8 Step 4(对比 + 正确性) |
| §6 flushHostCache | Task 4(写入后 / 读取前均有) |
| §6 三段时间分开报告 | Task 4、Task 7 的报告块 |
| §6 核心指标(链路流量/带宽) | Task 4、Task 7 报告块;Task 8 出表 |
| §6 测量纪律(≥5 次取最小、warmup、扫 batchSize) | Task 4(reps/warmup)、Task 5(扫描) |
| §7 正确性验证(逐位一致) | Task 4 Step 3、Task 7 Step 3/4 |
| §7 内建 hostPrintf 原始比特调试 | Task 3、Task 6 的 kernel 均含 |
| §8 文件组织 | Task 1 |
| §9 交付顺序 | Task 1→8 即该顺序 |
| §10 风险对策 | 各任务对应步骤已含 |

**类型一致性检查:** kernel 符号名 `hamming_scan_dists` / `hamming_scan_hist` /
`hamming_merge_topk` 在 Task 3、6(定义)与 Task 4、7(`createFunction`)中一致;
`outMeta` 布局 `[candCount, overflow]` 在 Task 6 定义、Task 7 消费处一致;
`WORDS=4` / `BITS=256` / `NBUCKETS=257` 在 kernel、host、python 三侧一致。
