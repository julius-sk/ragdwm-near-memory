#!/usr/bin/env bash
# fetch.sh —— 下载并整理测试数据集。地址全部在 2026-10-23 用 HTTP HEAD 核对过
# (见 RAGDWM/research_notes/h1_workload_datasets.md),可续传(curl -C -)。
#
# 为什么选这几个(都是相关论文实际用过的,数字可以直接对照):
#   gist1m       960 维 100 万条,自带 k=100 GT。RaBitQ(SIGMOD'24)、Extended RaBitQ、
#                GPU IVF-RaBitQ、DiskANN 都用 —— 经典小基准,和 RaBitQ 论文逐项比
#   openai3072   DBpedia 100 万条 text-embedding-3-large(3072 维),Extended RaBitQ 与
#                GPU IVF-RaBitQ 用过。没有查询,留出 1,000 条当查询,GT 自己算
#   openai1536   同一批文档的 ada-002 1536 维(在同一份 parquet 里,不用另下)
#   deep         DEEP1B,96 维,CXL-ANNS、DiskANN、SPANN、REIS、big-ann 都用。
#                官方 10K 查询,10M / 100M / 1B 三档官方 GT。10M / 100M 只下载
#                base.1B.fbin 的前一段(big-ann 的裁剪就是取前 N 条)
#   beir-nq / beir-hotpotqa / beir-msmarco
#                Cohere embed-english-v3(1024 维)+ 人工相关性标注(qrels)。REIS 用了
#                NQ 和 HotpotQA。这一组报 nDCG@10,是 RAG 真正关心的指标
#
# 用法:bash fetch.sh <数据集> [数据目录,默认 $HOME/rbq_data]
#   数据集:gist1m | openai | deep10m | deep100m | deep1b | beir-nq | beir-hotpotqa | beir-msmarco
#   实验室网络有 TLS 中间人,证书报错时:CURL_K=1 bash fetch.sh ...
set -euo pipefail
DS="${1:?用法: bash fetch.sh <数据集> [数据目录]}"
ROOT="${2:-$HOME/rbq_data}"
cd "$(dirname "$0")"
HERE="$PWD"
K=""; [ "${CURL_K:-0}" = 1 ] && K="-k"
mkdir -p "$ROOT/raw"
dl() {   # dl <url> <目标文件> [字节数上限]
    local url="$1" out="$2" lim="${3:-}"
    if [ -n "$lim" ]; then
        local have=0
        [ -f "$out" ] && have=$(stat -c %s "$out")
        if [ "$have" -ge "$lim" ]; then echo "已有 $out"; return; fi
        echo "下载 $url 的前 $lim B -> $out"
        curl $K -fL -r "$have-$((lim - 1))" "$url" >> "$out"
    else
        echo "下载 $url -> $out"
        curl $K -fL -C - -o "$out" "$url"
    fi
}
hf_dir() {   # hf_dir <仓库> <仓库内目录> <本地目录>:按 HF API 列出的文件逐个下载
    local repo="$1" sub="$2" dst="$3"
    mkdir -p "$dst"
    curl $K -fsL "https://huggingface.co/api/datasets/$repo/tree/main/$sub" \
        | python3 -c "import sys,json; [print(x['path']) for x in json.load(sys.stdin) if x['type']=='file']" \
        | while read -r p; do
            dl "https://huggingface.co/datasets/$repo/resolve/main/$p" "$dst/$(basename "$p")"
        done
}

YDX=https://storage.yandexcloud.net/yandex-research/ann-datasets
FB=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
case "$DS" in
gist1m)
    D="$ROOT/raw/gist"; mkdir -p "$D"
    dl ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz "$D/gist.tar.gz"
    [ -f "$D/gist/gist_base.fvecs" ] || tar -xzf "$D/gist.tar.gz" -C "$D"
    ls -l "$D/gist/"
    ;;
openai)
    D="$ROOT/raw/openai3072"
    hf_dir Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M data "$D"
    python3 "$HERE/hf_convert.py" openai --src "$D" --dim 3072 --out "$ROOT/openai3072"
    python3 "$HERE/hf_convert.py" openai --src "$D" --dim 1536 --out "$ROOT/openai1536"
    ;;
deep10m|deep100m|deep1b)
    D="$ROOT/raw/deep"; mkdir -p "$D"
    dl "$YDX/DEEP/query.public.10K.fbin" "$D/query.public.10K.fbin"
    case "$DS" in
    deep10m)  N=10000000;   dl "$FB/GT_10M/deep-10M" "$D/deep-10M.gt" ;;
    deep100m) N=100000000;  dl "$FB/GT_100M/deep-100M" "$D/deep-100M.gt" ;;
    deep1b)   N=1000000000; dl "$YDX/deep_new_groundtruth.public.10K.bin" "$D/deep-1B.gt" ;;
    esac
    # 只要前 N 条:8 B 头 + N x 96 x 4 B。头里的条数仍是 10 亿,vecio 以文件长度为准
    dl "$YDX/DEEP/base.1B.fbin" "$D/base.1B.fbin" $((8 + N * 384))
    ls -l "$D"
    ;;
beir-nq|beir-hotpotqa|beir-msmarco)
    S="${DS#beir-}"
    SPLIT=test; [ "$S" = msmarco ] && SPLIT=dev
    D="$ROOT/raw/beir_$S"
    hf_dir CohereLabs/beir-embed-english-v3 "$S/corpus" "$D/corpus"
    mkdir -p "$D/queries" "$D/qrels"
    dl "https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/main/$S/queries/$SPLIT.parquet" "$D/queries/$SPLIT.parquet"
    dl "https://huggingface.co/datasets/CohereLabs/beir-embed-english-v3/resolve/main/$S/qrels/$SPLIT.parquet" "$D/qrels/$SPLIT.parquet"
    python3 "$HERE/hf_convert.py" beir --src "$D" --split "$SPLIT" --out "$ROOT/beir_$S"
    ;;
*) echo "不认识的数据集 $DS"; exit 1 ;;
esac
echo "完成:$DS"
