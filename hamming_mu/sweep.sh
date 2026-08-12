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
         | grep 'scan (min' | awk '{print $6}')
    printf "%-12s %-12s\n" "$b" "${ms:-FAIL}"
done

echo
echo "=== numSub sweep (batchSize 用上面的最优值,手工填入 BEST_B) ==="
BEST_B="${BEST_B:-4}"
printf "%-12s %-12s\n" "numSub" "scan_ms"
for s in 4 8 16 22; do
    ms=$(./hamming_scan -n "$N" -t "$T" -b "$BEST_B" -s "$s" --reps 5 2>/dev/null \
         | grep 'scan (min' | awk '{print $6}')
    printf "%-12s %-12s\n" "$s" "${ms:-FAIL}"
done
