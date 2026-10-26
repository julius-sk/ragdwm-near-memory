#!/usr/bin/env bash
# interfere.sh —— 干扰实验:检索和主机上的内存密集任务同时跑,双方各慢多少。
#
# 近内存方案的主张之一是「检索不占主机」。单独比检索延迟证明不了这一点。这里让
# stream_triad(代表主机上的其他工作,例如 CPU 上的推理前后处理)和检索并发:
#   A  stream_triad 单独                      -> 主机可用带宽
#   B  host_rabitq 单独                       -> 主机检索延迟
#   C  stream_triad + host_rabitq 并发         -> 两者互相拖慢多少
#   D  rabitq_scan(卡)单独
#   E  stream_triad + rabitq_scan(卡)并发     -> 卡上检索对主机带宽的影响,以及主机繁忙对卡的影响
#
# 用法:bash interfere.sh <编码前缀,例如 ~/rbq_data/enc/deep100m_1b>
#   STREAM_THREADS=96 STREAM_GB=16 SECONDS_=40 NQ=200
set -uo pipefail
P="${1:?用法: bash interfere.sh <编码前缀>}"
cd "$(dirname "$0")"
B="$PWD"; RM="$(cd .. && pwd)"
ST="${STREAM_THREADS:-$(( $(nproc) / 4 ))}"; SG="${STREAM_GB:-16}"; SEC="${SECONDS_:-40}"; NQ="${NQ:-200}"
NB=$(python3 -c "import numpy as np;print(int(np.fromfile('$P.meta','<u8',8)[4]))")
M=G; [ "$NB" -eq 1 ] && M=T
OUT="$B/results/interfere_$(basename "$P")_$(date +%Y%m%d_%H%M)"; mkdir -p "$OUT"
log() { echo "$*" | tee -a "$OUT/summary.txt"; }
stream_med() { grep "GB/s" "$1" | awk '{print $(NF-1)}' | sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
log "前缀 $P  nb=$NB  stream 线程 $ST / 共 $(nproc)  stream 数组 $SG GB"

OMP_NUM_THREADS=$ST "$B/stream_triad" --gb "$SG" --seconds "$SEC" > "$OUT/A_stream.txt"
log "A  stream 单独:           $(stream_med "$OUT/A_stream.txt") GB/s"

numactl --interleave=all "$B/host_rabitq" --data "$P" --nq "$NQ" --reps 1 > "$OUT/B_host.txt" 2>&1
log "B  主机检索单独:         $(grep 每批延迟 "$OUT/B_host.txt")"

OMP_NUM_THREADS=$ST "$B/stream_triad" --gb "$SG" --seconds "$SEC" > "$OUT/C_stream.txt" &
SP=$!; sleep 5
numactl --interleave=all "$B/host_rabitq" --data "$P" --nq "$NQ" --reps 1 > "$OUT/C_host.txt" 2>&1
wait $SP
log "C  并发:stream            $(stream_med "$OUT/C_stream.txt") GB/s(有一部分时间检索还没开始或已结束,看 C_stream.txt 逐秒数字)"
log "   并发:主机检索         $(grep 每批延迟 "$OUT/C_host.txt")"

(cd "$RM" && ./rabitq_scan --data "$P" --mode "$M" --nq "$NQ" --reps 1) > "$OUT/D_dev.txt" 2>&1
log "D  卡上检索单独:         $(grep -E '合计|扫描 kernel' "$OUT/D_dev.txt" | head -1)"

OMP_NUM_THREADS=$ST "$B/stream_triad" --gb "$SG" --seconds "$SEC" > "$OUT/E_stream.txt" &
SP=$!; sleep 5
(cd "$RM" && ./rabitq_scan --data "$P" --mode "$M" --nq "$NQ" --reps 1) > "$OUT/E_dev.txt" 2>&1
wait $SP
log "E  并发:stream            $(stream_med "$OUT/E_stream.txt") GB/s"
log "   并发:卡上检索         $(grep -E '合计|扫描 kernel' "$OUT/E_dev.txt" | head -1)"
log ""
log "注意:卡上检索的一次性 fill / preload 发生在 rabitq_scan 启动时,会占主机内存带宽和 CXL 链路;"
log "E 的 stream 逐秒数字里能看到这一段。稳态比较请看检索进入查询循环之后的那几秒。"
log "完成。把 $OUT/summary.txt 发回来。"
