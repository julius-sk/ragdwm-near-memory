#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_scale.sh —— solab-h1 上的规模验证:MS MARCO 884 万段落,10 万 / 100 万 / 全部。
#
# 检验开发机四个小语料(最大 5.7 万条)上的结论在大一到两个数量级时是否成立:
#   1. 保住精确 top-10 所需候选数:SimHash 随库增大,RaBitQ 几乎不变
#   2. 两阶段精算比例:FAISS 单线程 vs 2816 task 各自堆 vs 2816 task 全局 τ
#
# 前提:pip install numpy faiss-cpu numba torch sentence-transformers
#       有 GPU(A100)编码约半小时;没有就走 CPU,慢得多但可续跑
#
# 用法:
#   bash run_scale.sh                       # 下载(若缺)-> 编码 -> 统计
#   BEIR_DIR=/data/msmarco bash run_scale.sh
#   DEVICE=1 bash run_scale.sh              # 另外在卡上跑 G/T 模式(需 ../rabitq_scan 已构建并通过 run_rabitq.sh)
#   SIZES=100000,1000000 bash run_scale.sh  # 先跑小的
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")"

BEIR_DIR="${BEIR_DIR:-$PWD/beir/msmarco}"
EMB_DIR="${EMB_DIR:-$PWD/emb_msmarco}"
SIZES="${SIZES:-100000,1000000,0}"
NQ="${NQ:-1000}"
NQ_REFINE="${NQ_REFINE:-100}"
DEVICE="${DEVICE:-0}"
OUT="$PWD/results_$(date +%Y%m%d_%H%M)"      # 绝对路径:上卡那段会 cd ..
mkdir -p "$OUT"
log() { echo "$*" | tee -a "$OUT/summary.txt"; }

{
    echo "host  : $(hostname -s)   date: $(date -Is)"
    echo "git   : $(git rev-parse --short HEAD 2>/dev/null)"
    echo "config: sizes=$SIZES nq=$NQ nq_refine=$NQ_REFINE device=$DEVICE"
    python3 -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1
    python3 -c "import faiss,numba;print('faiss', faiss.__version__, 'numba', numba.__version__)" 2>&1
    free -g | head -2
} | tee "$OUT/00_provenance.txt"

# ---- 1. BEIR MS MARCO ----
if [ ! -f "$BEIR_DIR/corpus.jsonl" ]; then
    log "== 下载 BEIR msmarco(约 1 GB)"
    mkdir -p "$(dirname "$BEIR_DIR")"
    URL="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    # 实验室做 TLS 中间人劫持:先正常下,失败再 -k(下的是公开数据集,不执行)
    ( cd "$(dirname "$BEIR_DIR")" && { curl -fL -o msmarco.zip "$URL" || curl -kfL -o msmarco.zip "$URL"; } \
        && unzip -q -o msmarco.zip ) || { log "!! 下载失败。在别处下好 msmarco.zip,解压到 $BEIR_DIR 再重跑"; exit 1; }
fi
log "BEIR msmarco: $(wc -l < "$BEIR_DIR/corpus.jsonl") 段落"

# ---- 2. 编码(可续跑)----
log "== 编码 bge-small(被杀掉后重跑会从断点继续)"
python3 encode_msmarco.py --beir "$BEIR_DIR" --out "$EMB_DIR" 2>&1 | tee "$OUT/01_encode.txt" | grep -E "device|queries|corpus|块 .*(0|5)/|docs ->"
[ -f "$EMB_DIR/docs.npy" ] && [ -f "$EMB_DIR/queries.npy" ] || { log "!! 编码没完成"; exit 1; }

# ---- 3. 规模统计 ----
log "== 规模统计 sizes=$SIZES"
python3 scale_kprime.py --emb "$EMB_DIR" --sizes "$SIZES" --nq "$NQ" --nq-refine "$NQ_REFINE" \
    --out "$OUT/scale_results.json" 2>&1 | grep -v Warning | tee -a "$OUT/summary.txt"

# ---- 4.(可选)上卡:MS MARCO 真实嵌入上的两阶段 ----
if [ "$DEVICE" -eq 1 ]; then
    cd ..
    [ -x ./rabitq_scan ] || { log "!! 没有 ../rabitq_scan,先跑 run_rabitq.sh"; exit 1; }
    for n in 1000000 0; do
        tag=$([ "$n" -eq 0 ] && echo full || echo $((n/1000000))m)
        P="data/msmarco_${tag}_4b"
        [ -f "$P.meta" ] || python3 prep_rabitq.py --emb "$EMB_DIR" --subset "$n" --nb 4 \
            --nq 100 --out "$P" >> "$OUT/02_prep.txt" 2>&1
        for M in G T; do
            log "== 上卡 msmarco $tag nb=4 mode $M"
            ./rabitq_scan --data "$P" --mode "$M" --nq 100 --reps 1 --dump "$OUT/dev_${tag}_${M}.bin" \
                > "$OUT/dev_${tag}_${M}.txt" 2>&1 || { log "!! 执行失败"; continue; }
            grep -E "第一遍|扫描 kernel|τ\+|归并 kernel|合计|精算|链路" "$OUT/dev_${tag}_${M}.txt" | tee -a "$OUT/summary.txt"
            python3 verify_rabitq.py --data "$P" --mode "$M" --dump "$OUT/dev_${tag}_${M}.bin" --nq 20 \
                2>&1 | tail -3 | tee -a "$OUT/summary.txt"
        done
    done
    cd scale
fi

log ""
log "完成。把 $OUT/summary.txt 和 $OUT/scale_results.json 发回来。"
