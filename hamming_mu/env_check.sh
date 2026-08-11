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
