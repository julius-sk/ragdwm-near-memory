#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_paradigm_e.sh —— 范式 E(近内存 near-memory)的判决性测量,一次跑完。
#
# 回答的问题只有一个:27.9 / 35.3 GB/s 的设备扫描,是撞上了硬件带宽墙,
# 还是撞上了我们自己的实现?
#
# 为什么要写成脚本而不是列一串命令:这六个测量必须在**同一个配置、同一次
# 会话**里完成才能相互比较。RESULTS.md 里的设备数字取自 8 月,而 bench_host_c.c
# 之后修过 NUMA 首次触碰(first-touch),hamming_scan.cpp 加过 mode R —— 拿今天
# 的 roofline 去比 8 月的扫描,比值没有意义。
#
# 全部输出落在 results_<时间戳>/ 下,原始文件一个不删。
#
# 用法:
#   bash run_paradigm_e.sh                # 默认 N=100000000,与 RESULTS.md 同口径
#   N=10000000 bash run_paradigm_e.sh     # 快速冒烟(10M,约 1/10 时间)
#   SKIP_BUILD=1 bash run_paradigm_e.sh   # 已经编好了
# ---------------------------------------------------------------------------
set -uo pipefail

N="${N:-100000000}"
TASKS="${TASKS:-2816}"
BATCH="${BATCH:-16}"
NUMSUB="${NUMSUB:-8}"
REPS="${REPS:-5}"
# 三个工具的默认 topK 都是 500,8 月的设备数字也是在默认值下取的。改动它会让
# 主机侧和设备侧的 top-k 维护成本不再可比 —— 除非同时改,否则别动。
K="${K:-500}"
SKIP_BUILD="${SKIP_BUILD:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
OUT="results_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT" || { echo "无法创建 $OUT"; exit 1; }

# tee 到文件的同时保留屏幕输出;所有小节都用它,便于事后复查。
run() {  # run <输出文件> <说明> <命令...>
    local f="$OUT/$1"; local what="$2"; shift 2
    echo
    echo "=============================================================="
    echo "== $what"
    echo "==   $*"
    echo "=============================================================="
    { printf '$ %s\n\n' "$*"; } > "$f"
    "$@" 2>&1 | tee -a "$f"
    local rc="${PIPESTATUS[0]}"
    echo "  [exit $rc]" | tee -a "$f"
    return "$rc"
}

echo "范式 E 判决测量   N=$N  t=$TASKS  b=$BATCH  s=$NUMSUB  reps=$REPS"
echo "输出目录: $OUT"
{
    echo "host      : $(hostname -s)"
    echo "date      : $(date -Is)"
    echo "kernel    : $(uname -r)"
    echo "config    : N=$N tasks=$TASKS batch=$BATCH numSub=$NUMSUB reps=$REPS k=$K"
    echo "git       : $(git rev-parse --short HEAD 2>/dev/null || echo '(非 git 工作区)')"
    echo "git dirty : $(git status --porcelain 2>/dev/null | wc -l) 个文件有未提交改动"
} | tee "$OUT/00_provenance.txt"

# ---- 0. 环境 --------------------------------------------------------------
run 01_env.txt "环境校验" bash env_check.sh

# ---- 1. 构建 --------------------------------------------------------------
if [ "$SKIP_BUILD" -eq 0 ]; then
    run 02_build.txt "构建(含 mu_roofline.cpp 与 bench_roofline)" bash build.sh \
        || { echo; echo "!! 构建失败,后面的测量都不用跑了。看 $OUT/02_build.txt"; exit 1; }
else
    echo "(SKIP_BUILD=1,跳过构建)"
fi

for exe in hamming_scan bench_host_c bench_roofline; do
    [ -x "./$exe" ] || echo "!! 缺少可执行文件 ./$exe —— 相关小节会失败"
done

# ---- 2. 向量化闸门 --------------------------------------------------------
# 这一步放在任何主机数字之前。如果编译器没有向量化,"分块扫描"那一行就只是
# 另一个标量实现,拿它去和设备比是错的,而错法很隐蔽:数字看起来完全正常。
echo
echo "=============================================================="
echo "== 向量化闸门:bench_roofline 里有多少条 VPOPCNTDQ"
echo "=============================================================="
VPOP="$(objdump -d ./bench_roofline 2>/dev/null | grep -c vpopcnt || echo 0)"
POPC="$(objdump -d ./bench_roofline 2>/dev/null | grep -cE '\bpopcnt\b' || echo 0)"
{
    echo "vpopcnt (AVX-512 VPOPCNTDQ) : $VPOP"
    echo "popcnt  (标量)              : $POPC"
    if [ "$VPOP" -eq 0 ]; then
        echo
        echo "!! 为 0:编译器没有向量化。"
        echo "   '分块扫描' 那一行不能用作结论 —— 它只是第二个标量实现。"
        echo "   检查 -march=native 是否生效,以及这颗 CPU 是否支持 AVX512_VPOPCNTDQ:"
        echo "     grep -o 'avx512_vpopcntdq' /proc/cpuinfo | head -1"
        echo "   Granite Rapids 支持,所以为 0 基本是编译选项问题,不是硬件问题。"
    else
        echo "   向量化已发生,可以采信分块扫描的数字。"
    fi
} | tee "$OUT/03_vectorization_gate.txt"

# ---- 3. 主机侧:roofline / 标量扫描 / 分块扫描 ----------------------------
run 04_host_roofline.txt "主机 roofline + 标量扫描 + 分块扫描" \
    ./bench_roofline -n "$N" -k "$K" --reps "$REPS"

# bench_host_c 必须重跑:NUMA 首次触碰修过之后,RESULTS.md 里 51.8 GB/s 那个
# 数字已经不代表当前代码了。
run 05_host_scan_c.txt "主机 A 组基线(NUMA 修正后重测)" \
    ./bench_host_c -n "$N" -k "$K" --reps "$REPS"

# ---- 4. 设备侧:roofline ---------------------------------------------------
run 06_device_roofline.txt "设备 roofline(mode R,纯读 / 读+写)" \
    ./hamming_scan -n "$N" -t "$TASKS" -b "$BATCH" -s "$NUMSUB" --mode R --reps "$REPS"

# ---- 5. 设备侧:B 组与 C 组,同配置重测 ------------------------------------
# 和 roofline 同一次会话、同一份固件、同一个 SDK。只有这样比值才成立。
run 07_device_scan_B.txt "设备扫描 B 组(同配置重测)" \
    ./hamming_scan -n "$N" -t "$TASKS" -b "$BATCH" -s "$NUMSUB" --mode B --reps "$REPS"

run 08_device_scan_C.txt "设备扫描 C 组(同配置重测)" \
    ./hamming_scan -n "$N" -t "$TASKS" -b "$BATCH" -s "$NUMSUB" --mode C --reps "$REPS"

# ---- 6. 汇总 --------------------------------------------------------------
# 抽取用 grep,取不到就留空 —— 宁可让表格有洞,也不要把解析错误当成测量结果。
num() { grep -oE '[0-9]+\.[0-9]+' | sed -n "${1}p"; }

HOST_RF="$(grep '纯顺序读'   "$OUT/04_host_roofline.txt" 2>/dev/null | num 2)"
HOST_SC="$(grep '标量扫描'   "$OUT/04_host_roofline.txt" 2>/dev/null | num 2)"
HOST_BL="$(grep '分块扫描'   "$OUT/04_host_roofline.txt" 2>/dev/null | num 2)"
DEV_RF="$(grep '纯读 (roofline)' "$OUT/06_device_roofline.txt" 2>/dev/null | num 2)"
DEV_RW="$(grep '读+写'       "$OUT/06_device_roofline.txt" 2>/dev/null | num 2)"
DEV_B="$(grep '扫描有效带宽' "$OUT/07_device_scan_B.txt" 2>/dev/null | num 1)"
DEV_C="$(grep '扫描有效带宽' "$OUT/08_device_scan_C.txt" 2>/dev/null | num 1)"

pct() { [ -n "${1:-}" ] && [ -n "${2:-}" ] && awk -v a="$1" -v b="$2" \
        'BEGIN{ if (b>0) printf "%.1f%%", 100*a/b; else print "n/a" }' || echo "n/a"; }

{
    echo "==================================================================="
    echo " 范式 E 判决表    N=$N   t=$TASKS b=$BATCH s=$NUMSUB"
    echo " $(hostname -s)   $(date -Is)"
    echo "==================================================================="
    printf "%-32s %12s %14s\n" "" "GB/s" "占自身 roofline"
    printf "%-32s %12s %14s\n" "主机 roofline(纯读)"      "${HOST_RF:-n/a}" "100%"
    printf "%-32s %12s %14s\n" "主机扫描(标量)"           "${HOST_SC:-n/a}" "$(pct "${HOST_SC:-}" "${HOST_RF:-}")"
    printf "%-32s %12s %14s\n" "主机扫描(分块/向量化)"    "${HOST_BL:-n/a}" "$(pct "${HOST_BL:-}" "${HOST_RF:-}")"
    printf "%-32s %12s %14s\n" "设备 roofline(纯读)"      "${DEV_RF:-n/a}"  "100%"
    printf "%-32s %12s %14s\n" "设备 roofline(读+写)"     "${DEV_RW:-n/a}"  "$(pct "${DEV_RW:-}" "${DEV_RF:-}")"
    printf "%-32s %12s %14s\n" "设备扫描 B"               "${DEV_B:-n/a}"   "$(pct "${DEV_B:-}" "${DEV_RF:-}")"
    printf "%-32s %12s %14s\n" "设备扫描 C"               "${DEV_C:-n/a}"   "$(pct "${DEV_C:-}" "${DEV_RF:-}")"
    echo
    echo "8 月记录(仅供对照,口径可能已变):主机标量 51.8 / 设备 B 27.9 / 设备 C 35.3"
    echo
    echo "--- 判据(RUNBOOK 10.3,跑之前就定好的,不许事后改)---"
    if [ -n "${HOST_BL:-}" ] && [ -n "${DEV_RF:-}" ]; then
        awk -v h="$HOST_BL" -v d="$DEV_RF" 'BEGIN{
            if (h > d) print "判据 1 成立:主机向量化扫描 (" h ") > 设备 roofline (" d ")\n  => 结束近内存方向。主机写对就越过了设备的硬件极限,优化 kernel 不可能改变结论。"
            else print "判据 1 不成立:主机向量化扫描 (" h ") <= 设备 roofline (" d ")"
        }'
    else
        echo "判据 1:数据缺失,无法判定"
    fi
    if [ -n "${DEV_C:-}" ] && [ -n "${DEV_RF:-}" ]; then
        awk -v c="$DEV_C" -v d="$DEV_RF" 'BEGIN{
            r = (d>0) ? 100*c/d : 0
            if (r > 85) printf "判据 2 成立:设备扫描 C 占 roofline %.1f%% (>85%%)\n  => 结束。设备已贴着自己的天花板,缺口不在实现。\n", r
            else printf "判据 2 不成立:设备扫描 C 占 roofline %.1f%% (<=85%%)\n  => 缺口在实现,值得继续。下一步定位到具体原因(访存模式/batchSize/numSub/指令调度),不要再泛泛优化。\n", r
        }'
    else
        echo "判据 2:数据缺失,无法判定"
    fi
    echo
    echo "校验提醒:"
    echo "  - mode R 输出里的 'XOR 折叠' 为 0 说明 kernel 可能根本没跑,此时带宽数字无效"
    echo "  - VPOPCNTDQ 计数 = $VPOP,为 0 则'主机扫描(分块)'一行不成立"
    echo "  - bench_roofline 自带 top-k 一致性校验,不通过它会拒绝报告性能数字"
} | tee "$OUT/09_verdict.txt"

echo
echo "完成。全部原始输出在 $OUT/"
echo "把 $OUT/09_verdict.txt 和 $OUT/0[4-8]_*.txt 发回来即可。"
