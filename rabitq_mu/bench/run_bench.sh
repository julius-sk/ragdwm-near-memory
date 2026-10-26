#!/usr/bin/env bash
# run_bench.sh —— 一个数据集跑完整套对比:编码 -> 地面真值 -> 卡 -> 主机 C -> FAISS -> 召回 / nDCG -> 重排。
#
# 用法:bash run_bench.sh <数据集> [数据目录,默认 $HOME/rbq_data]
#   数据集:gist1m | openai3072 | openai1536 | deep10m | deep100m | deep1b | beir_nq | beir_hotpotqa | beir_msmarco
#   先用 fetch.sh 下好数据。
# 环境变量:
#   NBS="1 2 4"     每维位数(1 = 只有符号码)
#   NQ_DEV=1000     卡上和 FAISS 批量跑的查询数
#   NQ_HOST=100     主机 C 单查询延迟的查询数(大库上单查询要几百毫秒)
#   PRELOAD_GB=     h1_probe.sh 量出的最大连续 preload(GB);卡上所需超过它就跳过卡
#   TASKS=2816 BATCH=16 NUMSUB=8   卡的并行参数(hamming_mu 在 h1 上扫出来的最优值,RaBitQ 未必最优)
#   SKIP_DEVICE=1   只跑主机部分(比如在没有卡的机器上)
#
# 每一步的输出都在 results/<数据集>_<时间>/,summary.txt 是汇总。设备与主机的 top-k 逐 id 对照,
# 对不上就停下来 —— 一个快但错的结果比没有结果更糟。
set -uo pipefail
DS="${1:?用法: bash run_bench.sh <数据集> [数据目录]}"
ROOT="${2:-$HOME/rbq_data}"
cd "$(dirname "$0")"
B="$PWD"; RM="$(cd .. && pwd)"
NBS="${NBS:-1 2 4}"; NQ_DEV="${NQ_DEV:-1000}"; NQ_HOST="${NQ_HOST:-100}"
TASKS="${TASKS:-2816}"; BATCH="${BATCH:-16}"; NUMSUB="${NUMSUB:-8}"
OUT="$B/results/${DS}_$(date +%Y%m%d_%H%M)"; mkdir -p "$OUT" "$ROOT/enc" "$ROOT/gt"
log() { echo "$*" | tee -a "$OUT/summary.txt"; }
py() { python3 "$@"; }

# ---- 数据集表:基础向量、查询、裁剪、GT 来源 ----
CROP=0; QKEY=test; OFFICIAL_GT=""; CHECK_GT=""; CONV=""; EXTRA=""
case "$DS" in
gist1m)  BASE="$ROOT/raw/gist/gist/gist_base.fvecs"; QUERY="$ROOT/raw/gist/gist/gist_query.fvecs"
         CHECK_GT="$ROOT/raw/gist/gist/gist_groundtruth.ivecs" ;;
openai3072|openai1536) BASE="$ROOT/$DS/docs.npy"; QUERY="$ROOT/$DS/queries.npy" ;;
deep10m)  BASE="$ROOT/raw/deep/base.1B.fbin"; QUERY="$ROOT/raw/deep/query.public.10K.fbin"
          CROP=10000000; OFFICIAL_GT="$ROOT/raw/deep/deep-10M.gt" ;;
deep100m) BASE="$ROOT/raw/deep/base.1B.fbin"; QUERY="$ROOT/raw/deep/query.public.10K.fbin"
          CROP=100000000; OFFICIAL_GT="$ROOT/raw/deep/deep-100M.gt" ;;
deep1b)   BASE="$ROOT/raw/deep/base.1B.fbin"; QUERY="$ROOT/raw/deep/query.public.10K.fbin"
          CROP=1000000000; OFFICIAL_GT="$ROOT/raw/deep/deep-1B.gt" ;;
beir_nq|beir_hotpotqa|beir_msmarco)
          BASE="$ROOT/$DS/docs.npy"; QUERY="$ROOT/$DS/queries.npy"; CONV="$ROOT/$DS"
          EXTRA="--normalized-ip" ;;
*) echo "不认识的数据集 $DS"; exit 1 ;;
esac
[ -e "$BASE" ] || { echo "没有 $BASE —— 先跑 bash fetch.sh"; exit 1; }

{
    echo "host  : $(hostname -s)   date: $(date -Is)"
    echo "git   : $(git -C "$RM" rev-parse --short HEAD 2>/dev/null)"
    echo "data  : $DS  base=$BASE crop=$CROP"
    echo "config: nbs=[$NBS] nq_dev=$NQ_DEV nq_host=$NQ_HOST tasks=$TASKS batch=$BATCH numSub=$NUMSUB preload_gb=${PRELOAD_GB:-?}"
    echo "cpu   : $(lscpu | grep 'Model name' | sed 's/.*: *//') x $(nproc) 线程;内存 $(free -g | awk '/Mem/{print $2}') GB"
} | tee "$OUT/00_provenance.txt"

# ---- 地面真值 ----
GT="$ROOT/gt/${DS}.gt100.bin"
if [ -n "$OFFICIAL_GT" ]; then
    GT="$OFFICIAL_GT"; log "GT:官方 $GT"
elif [ ! -f "$GT" ]; then
    log "GT:计算精确 top-100 -> $GT"
    py "$B/gt.py" --base "$BASE" --crop "$CROP" --queries "$QUERY" --k 100 --out "$GT" \
        ${CHECK_GT:+--check "$CHECK_GT"} > "$OUT/01_gt.txt" 2>&1 || { log "!! GT 失败"; exit 1; }
    grep "官方" "$OUT/01_gt.txt" | tee -a "$OUT/summary.txt"
fi
if [ -n "$CONV" ]; then
    py "$B/ir_eval.py" --conv "$CONV" --gt "$GT" --k 10 --json "$OUT/results.jsonl" \
        | sed 's/^/  精确搜索(上限) /' | tee -a "$OUT/summary.txt"
fi

for NB in $NBS; do
    P="$ROOT/enc/${DS}_${NB}b"; TAG="${DS}_${NB}b"
    log ""; log "======== $TAG ========"
    if [ ! -f "$P.info.json" ]; then
        py "$B/encode.py" --base "$BASE" --crop "$CROP" --queries "$QUERY" --nb "$NB" --k 10 \
            $EXTRA --out "$P" > "$OUT/${TAG}_encode.txt" 2>&1 || { log "!! 编码失败,见 ${TAG}_encode.txt"; exit 1; }
    fi
    tail -1 "$OUT/${TAG}_encode.txt" 2>/dev/null | tee -a "$OUT/summary.txt"
    NEED_GB=$(py -c "import json;i=json.load(open('$P.info.json'));print(int((i['bytes_stage1']+i['bytes_stage2']+i['bytes_lb'])/1e9)+2)")
    M=G; [ "$NB" -eq 1 ] && M=T

    # ---- 卡 ----
    DEV="$OUT/${TAG}_dev.bin"
    if [ "${SKIP_DEVICE:-0}" = 1 ]; then
        log "  卡:跳过(SKIP_DEVICE=1)"; DEV=""
    elif [ -n "${PRELOAD_GB:-}" ] && [ "$NEED_GB" -gt "$PRELOAD_GB" ]; then
        log "  卡:需要约 $NEED_GB GB preload,超过本卡最大连续块 $PRELOAD_GB GB —— 跳过"; DEV=""
    else
        (cd "$RM" && ./rabitq_scan -t "$TASKS" -b "$BATCH" -s "$NUMSUB" --data "$P" --mode "$M" \
            --nq "$NQ_DEV" --reps 2 --dump "$DEV") > "$OUT/${TAG}_dev.txt" 2>&1 \
            || { log "!! 卡上执行失败,见 ${TAG}_dev.txt"; DEV=""; }
        [ -n "$DEV" ] && grep -E "中位|有效带宽|精算|链路回传" "$OUT/${TAG}_dev.txt" | sed 's/^/  卡 /' | tee -a "$OUT/summary.txt"
    fi
    if [ -n "$DEV" ]; then
        py "$B/eval.py" --dump "$DEV" --gt "$GT" --k 10 --json "$OUT/results.jsonl" | sed 's/^/  卡 /' | tee -a "$OUT/summary.txt"
        [ -n "$CONV" ] && py "$B/ir_eval.py" --conv "$CONV" --dump "$DEV" --k 10 --json "$OUT/results.jsonl" | sed 's/^/  卡 /' | tee -a "$OUT/summary.txt"
    fi

    # ---- 主机 C:单查询延迟 + 批量吞吐 ----
    HOST="$OUT/${TAG}_host.bin"
    numactl --interleave=all "$B/host_rabitq" --data "$P" --qbatch 1 --nq "$NQ_HOST" --reps 2 --dump "$HOST" \
        > "$OUT/${TAG}_host_b1.txt" 2>&1 || log "!! 主机 C 失败,见 ${TAG}_host_b1.txt"
    grep -E "每批延迟|吞吐|带宽|精算" "$OUT/${TAG}_host_b1.txt" | sed 's/^/  主机C(单查询) /' | tee -a "$OUT/summary.txt"
    numactl --interleave=all "$B/host_rabitq" --data "$P" --qbatch 64 --nq "$NQ_DEV" --reps 1 \
        > "$OUT/${TAG}_host_b64.txt" 2>&1 || log "!! 主机 C 批量失败"
    grep -E "每批延迟|吞吐" "$OUT/${TAG}_host_b64.txt" | sed 's/^/  主机C(64 一批) /' | tee -a "$OUT/summary.txt"
    py "$B/eval.py" --dump "$HOST" --gt "$GT" --k 10 | grep recall | sed 's/^/  主机C /' | tee -a "$OUT/summary.txt"
    if [ -n "$DEV" ]; then
        X=$(py "$B/eval.py" --dump "$DEV" --other "$HOST" --k 10 | grep "id 集合")
        log "  卡 vs 主机C:$X"
    fi

    # ---- FAISS(公开库)----
    numactl --interleave=all python3 "$B/faiss_host.py" --data "$P" --queries "$QUERY" --qkey "$QKEY" \
        --nq 20 --batch 1 > "$OUT/${TAG}_faiss_b1.txt" 2>&1
    grep "每批" "$OUT/${TAG}_faiss_b1.txt" | sed 's/^/  FAISS(单查询) /' | tee -a "$OUT/summary.txt"
    numactl --interleave=all python3 "$B/faiss_host.py" --data "$P" --queries "$QUERY" --qkey "$QKEY" \
        --nq "$NQ_DEV" --dump "$OUT/${TAG}_faiss.bin" --json "$OUT/results.jsonl" > "$OUT/${TAG}_faiss.txt" 2>&1
    grep "每批" "$OUT/${TAG}_faiss.txt" | sed 's/^/  FAISS(全部一批) /' | tee -a "$OUT/summary.txt"
    py "$B/eval.py" --dump "$OUT/${TAG}_faiss.bin" --gt "$GT" --k 10 | grep recall | sed 's/^/  FAISS /' | tee -a "$OUT/summary.txt"

    # ---- 漏斗最后一级:卡返回 64 个候选,主机读原始向量重排 ----
    if [ -n "$DEV" ] && [ "$NB" -le 2 ]; then
        P64="${P}_k64"
        [ -f "$P64.qblocks" ] || py "$B/encode.py" --base "$BASE" --crop "$CROP" --queries "$QUERY" \
            --k 64 --qblocks-only "$P" --out "$P64" > /dev/null
        (cd "$RM" && ./rabitq_scan -t "$TASKS" -b "$BATCH" -s "$NUMSUB" --data "$P64" --mode "$M" \
            --nq "$NQ_DEV" --reps 1 --dump "$OUT/${TAG}_dev_k64.bin") > "$OUT/${TAG}_dev_k64.txt" 2>&1
        py "$B/rescore.py" --dump "$OUT/${TAG}_dev_k64.bin" --kp 64 --base "$BASE" --crop "$CROP" \
            --queries "$QUERY" --gt "$GT" --k 10 --time --json "$OUT/results.jsonl" | sed 's/^/  重排 /' | tee -a "$OUT/summary.txt"
    fi
done
log ""; log "完成。把 $OUT/summary.txt 发回来。"
