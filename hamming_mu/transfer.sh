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
