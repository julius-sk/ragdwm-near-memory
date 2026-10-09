#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_rabitq.sh —— solab-h1 上一次跑完:正确性(I -> E -> F -> T 分层)+ 扫描带宽。
#
# 前提:
#   pip install numpy faiss-cpu numba        (numba 可选,只加速参考模型)
#   嵌入文件:从开发机拷 ~/.cache/ragdwm/emb/*.npz 到 ./emb/(scp),
#            没有就只跑合成数据那一段
#
# 用法:
#   bash run_rabitq.sh                 # 正确性 + 带宽,全部
#   SKIP_BUILD=1 bash run_rabitq.sh
#   BW_N=100000000 bash run_rabitq.sh  # 带宽测试的合成规模(默认 1 亿条)
#
# 排查顺序是设计好的,别跳:
#   I 不对 -> 位运算或数据布局错了(与浮点无关)
#   I 对、E 不对 -> 第一阶段的标量浮点在设备上算得不一样
#   E 对、F 不对 -> 第二阶段的 d 维标量浮点循环(DRAN 记录过 -O3 下算错)
#   F 对、T 不对 -> 堆、阈值、分块或归并逻辑
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

TASKS="${TASKS:-2816}"
BATCH="${BATCH:-16}"
NUMSUB="${NUMSUB:-8}"
NQ="${NQ:-50}"
BW_N="${BW_N:-100000000}"
OUT="results_$(date +%Y%m%d_%H%M)"
mkdir -p "$OUT" data
log() { echo "$*" | tee -a "$OUT/summary.txt"; }

{
    echo "host  : $(hostname -s)   date: $(date -Is)"
    echo "git   : $(git rev-parse --short HEAD 2>/dev/null) ($(git status --porcelain 2>/dev/null | wc -l) 个未提交改动)"
    echo "config: tasks=$TASKS batch=$BATCH numSub=$NUMSUB nq=$NQ bw_n=$BW_N"
} | tee "$OUT/00_provenance.txt"

bash env_check.sh > "$OUT/01_env.txt" 2>&1
if [ "${SKIP_BUILD:-0}" -eq 0 ]; then
    bash build.sh > "$OUT/02_build.txt" 2>&1 || { log "!! 构建失败,看 $OUT/02_build.txt"; exit 1; }
fi
[ -x ./rabitq_scan ] || { log "!! 没有 ./rabitq_scan"; exit 1; }

SCAN=(./rabitq_scan -t "$TASKS" -b "$BATCH" -s "$NUMSUB")

# ---- 正确性:每个语料 x 每个位数,按 I -> E -> F -> T ----
check() {   # check <prefix> <nb> <tag>
    local P="$1" NB="$2" TAG="$3"
    for M in I E F T; do
        [ "$M" = F ] && [ "$NB" -eq 1 ] && continue
        local D="$OUT/${TAG}_${M}.bin"
        "${SCAN[@]}" --data "$P" --mode "$M" --nq "$NQ" --reps 1 --dump "$D" \
            > "$OUT/${TAG}_${M}.txt" 2>&1 || { log "!! $TAG mode $M 执行失败(见 $OUT/${TAG}_${M}.txt)"; return 1; }
        python3 verify_rabitq.py --data "$P" --mode "$M" --dump "$D" --tasks "$TASKS" --nq "$NQ" \
            > "$OUT/${TAG}_${M}_verify.txt" 2>&1
        if [ $? -ne 0 ]; then
            log "!! $TAG mode $M 校验不通过 -> 停在这一层排查(见 $OUT/${TAG}_${M}_verify.txt)"
            return 1
        fi
        log "   $TAG mode $M 通过"
    done
}

shopt -s nullglob
EMBS=(emb/*.npz)
for E in "${EMBS[@]}"; do
    name=$(basename "$E" | cut -d_ -f1)
    for NB in 1 2 4; do
        P="data/${name}_${NB}b"
        [ -f "$P.meta" ] || python3 prep_rabitq.py --emb "$E" --nb "$NB" --nq "$NQ" --out "$P" \
            >> "$OUT/03_prep.txt" 2>&1
        log "== 正确性 $name nb=$NB"
        check "$P" "$NB" "${name}_${NB}b" || exit 1
    done
done
[ ${#EMBS[@]} -eq 0 ] && log "(emb/ 下没有 npz,跳过真实语料的正确性检查)"

# 合成数据也做一遍正确性(不依赖拷文件)
P="data/syn200k_4b"
[ -f "$P.meta" ] || python3 prep_rabitq.py --synthetic 200000 384 --nb 4 --nq "$NQ" --out "$P" >> "$OUT/03_prep.txt" 2>&1
log "== 正确性 合成 200k nb=4"
check "$P" 4 syn200k_4b || exit 1

# ---- 第一阶段扫描带宽:随机码,E 模式 ----
# 随机码不对应任何向量(1 亿条 x 384 维的浮点向量要 154 GB,超过 128 GB 主机内存),
# 所以只测第一阶段带宽;两阶段 T 的真实吞吐要在 MS MARCO 真实嵌入上测(scale/)。
# 第一阶段读 60 B/向量,与 nb 无关,所以只用 nb=1 的码。
P="data/rnd$((BW_N/1000000))m_1b"
[ -f "$P.meta" ] || python3 prep_rabitq.py --random-codes "$BW_N" 384 --nb 1 --nq 10 --out "$P" \
    >> "$OUT/03_prep.txt" 2>&1
log "== 第一阶段带宽 N=$BW_N(随机码,E 模式)"
"${SCAN[@]}" --data "$P" --mode E --nq 10 --reps 3 > "$OUT/bw_E.txt" 2>&1
grep -E "扫描 kernel|有效带宽" "$OUT/bw_E.txt" | tee -a "$OUT/summary.txt"
log "   对照:Hamming(hamming_mu C 组)32 B/向量 35.3 GB/s;这里 60 B/向量、每字 4 组 AND+popcount"

# batchSize 粗扫:本负载每字节的计算量约是 Hamming 的 4 倍,最优值未必还是 16
log "== batchSize 粗扫(E 模式, N=$BW_N)"
for B in 4 8 16 32; do
    ./rabitq_scan -t "$TASKS" -b "$B" -s "$NUMSUB" --data "$P" --mode E --nq 3 --reps 3 \
        | grep "扫描 kernel" | sed "s/^/   b=$B /" | tee -a "$OUT/summary.txt"
done

log ""
log "完成。全部输出在 $OUT/;把 $OUT/summary.txt 发回来即可。"
