"""encode_msmarco.py —— 用 bge-small 把 BEIR MS MARCO 编码成规模验证用的向量。

为什么用它而不是公开的现成嵌入:模型(BAAI/bge-small-en-v1.5,384 维)、查询前缀、
截断长度都和开发机上 SciFact / NFCorpus / ArguAna / FiQA 的实验一致,规模曲线才能
直接接上那四个小语料。

输入:BEIR 的 msmarco 目录(corpus.jsonl、queries.jsonl、qrels/dev.tsv)。
  下载:https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
  实验室网络做 TLS 中间人劫持,h1 上可能要 curl -kLO;也可以在别处下好再 scp。
输出(--out 目录):
  docs.npy      N x 384 float32(N = 8,841,823),以 memmap 逐块写入
  queries.npy   dev 集查询 x 384 float32(6,980 个)
  progress.txt  已完成的块号 —— 被杀掉后重跑只做剩下的块

用法:python encode_msmarco.py --beir /path/to/msmarco --out ../emb_msmarco
"""
import argparse
import json
import os
import time

import numpy as np

BGE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # 与 RAGDWM 的 BEIR 实验相同
CHUNK = 200_000


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beir", required=True, help="BEIR msmarco 目录")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=0, help="0 = GPU 256 / CPU 64")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = SentenceTransformer(BGE, device=dev)
    if dev == "cuda":
        m.half()
    m.max_seq_length = min(m.max_seq_length, 256)        # 与 RAGDWM 的 BEIR 实验相同
    batch = a.batch or (256 if dev == "cuda" else 64)
    print("device=%s batch=%d" % (dev, batch), flush=True)

    # ---- 查询:dev 集 ----
    qpath = os.path.join(a.out, "queries.npy")
    if not os.path.isfile(qpath):
        dev_ids = []
        for line in list(open(os.path.join(a.beir, "qrels", "dev.tsv"), encoding="utf-8"))[1:]:
            qid = line.split("\t")[0]
            if not dev_ids or dev_ids[-1] != qid:
                dev_ids.append(qid)
        dev_ids = list(dict.fromkeys(dev_ids))
        text = {}
        for line in open(os.path.join(a.beir, "queries.jsonl"), encoding="utf-8"):
            o = json.loads(line)
            text[o["_id"]] = o["text"]
        qv = m.encode([QUERY_PREFIX + text[q] for q in dev_ids], batch_size=batch,
                      normalize_embeddings=True, show_progress_bar=False)
        np.save(qpath, np.asarray(qv, np.float32))
        print("queries: %d -> %s" % (len(dev_ids), qpath), flush=True)

    # ---- 文档:逐块写进 memmap,可续跑 ----
    cpath = os.path.join(a.beir, "corpus.jsonl")
    dpath = os.path.join(a.out, "docs.npy")
    ppath = os.path.join(a.out, "progress.txt")
    N = count_lines(cpath)
    done = set(int(x) for x in open(ppath).read().split()) if os.path.isfile(ppath) else set()
    mm = (np.lib.format.open_memmap(dpath, mode="r+") if os.path.isfile(dpath)
          else np.lib.format.open_memmap(dpath, mode="w+", dtype=np.float32, shape=(N, 384)))
    assert mm.shape == (N, 384), "docs.npy 形状不对,删掉 docs.npy 与 progress.txt 重来"
    n_chunks = (N + CHUNK - 1) // CHUNK
    print("corpus: N=%d, %d 块, 已完成 %d 块" % (N, n_chunks, len(done)), flush=True)

    t0 = time.time()
    with open(cpath, encoding="utf-8") as f:
        for c in range(n_chunks):
            lines = [next(f) for _ in range(min(CHUNK, N - c * CHUNK))]
            if c in done:
                continue
            texts = []
            for line in lines:
                o = json.loads(line)
                texts.append(((o.get("title") or "") + " " + (o.get("text") or "")).strip())
            v = m.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False)
            mm[c * CHUNK:c * CHUNK + len(texts)] = np.asarray(v, np.float32)
            mm.flush()
            with open(ppath, "a") as p:
                p.write("%d\n" % c)
            el = time.time() - t0
            print("  块 %d/%d 完成,累计 %.0f s" % (c + 1, n_chunks, el), flush=True)
    print("docs -> %s" % dpath)


if __name__ == "__main__":
    main()
