#!/usr/bin/env bash
# h1_probe.sh —— 第 0 步:正式测试之前,把机器和卡的真实条件量清楚。只读,除了最后的 preload 探测。
#
# 量什么、为什么:
#   卡    固件 / Computing 模式 / InfiniteMemory 计数 —— 决定 MU 能不能算、SDK 是否配套
#   容量  Preload Region 槽位数和最大连续块 —— MU 只能读 Preload Region,库能放多大由它决定
#   主机  CPU 核数、内存、NUMA —— 主机基线的条件
#   GPU   有没有、型号、显存 —— 决定 GPU 基线在哪台机器上跑
#   磁盘  数据盘剩余空间与读带宽 —— DEEP1B 原始文件 384 GB,编码后还要几十 GB
#   占用  有没有别人在用卡 —— 实战手册 §12.4:别在别人用卡时跑大量写入
#
# 用法:bash h1_probe.sh [数据目录,默认 $HOME/rbq_data] > probe.txt 2>&1
#       PROBE_PRELOAD=0 bash h1_probe.sh    # 跳过 preload 探测(卡被别人占用时)
set -u
DATA="${1:-$HOME/rbq_data}"
cd "$(dirname "$0")"
sec() { echo; echo "=== $* ==="; }

sec "主机"
hostname; date -Is; uname -r
lscpu | grep -E "Model name|Socket|Core\(s\)|Thread|NUMA node\(s\)|L3"
free -g
numactl -H 2>/dev/null | head -12

sec "GPU"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
else
    echo "没有 nvidia-smi:这台机器跑不了 GPU 基线"
fi
python3 -c "import cuvs; print('cuvs', cuvs.__version__)" 2>/dev/null || echo "python 里没有 cuvs"

sec "XCENA 卡"
lspci -d 20a6: || echo "!! 没有 XCENA 设备"
sudo xcena_cli fw-info 0 2>/dev/null | head -20
xcena_cli device-info 0 2>/dev/null | head -30
daxctl list 2>/dev/null
sec "InfiniteMemory 计数(dram_capacity、used/dirty、nvme 读写)"
sudo xcena_cli im get-smart 0 2>/dev/null | head -40

sec "卡有没有被别人占用"
sudo fuser -v /dev/mx_dma/* /dev/dax* 2>&1 | grep -v "^$" || true
who

sec "数据盘 $DATA"
mkdir -p "$DATA"
df -h "$DATA"
findmnt -T "$DATA" -o SOURCE,FSTYPE,OPTIONS
DEV=$(findmnt -n -T "$DATA" -o SOURCE | sed 's/p\?[0-9]*$//' | xargs basename 2>/dev/null)
[ -n "$DEV" ] && sudo lspci -vv 2>/dev/null | grep -B30 -i "$DEV" | grep -E "LnkSta:" | head -2
if command -v fio >/dev/null; then
    fio --name=seqread --filename="$DATA/fio.tmp" --size=8G --rw=read --bs=1M --direct=1 \
        --ioengine=libaio --iodepth=32 --runtime=20 --time_based --group_reporting 2>/dev/null \
        | grep -E "READ:|bw="
    rm -f "$DATA/fio.tmp"
else
    echo "没有 fio(sudo apt install fio),跳过磁盘带宽"
fi

sec "Python 依赖"
python3 -c "import numpy, faiss; print('numpy', numpy.__version__, 'faiss', faiss.__version__, 'omp', faiss.omp_get_max_threads())"
python3 -c "import h5py; print('h5py ok')" 2>/dev/null || echo "没有 h5py(ann-benchmarks 的 hdf5 要用)"
python3 -c "import pyarrow; print('pyarrow ok')" 2>/dev/null || echo "没有 pyarrow(HuggingFace parquet 要用)"

if [ "${PROBE_PRELOAD:-1}" -eq 1 ]; then
    sec "Preload Region 容量探测"
    if [ -x ./preload_probe ]; then
        ./preload_probe --device 0
    else
        echo "先构建:bash build_bench.sh"
    fi
    sudo xcena_cli im get-smart 0 2>/dev/null | grep -E "dram|nvme" | head -10
fi
echo; echo "完成。把整个输出发回来。"
