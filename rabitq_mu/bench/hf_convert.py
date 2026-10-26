"""hf_convert.py —— 把 HuggingFace 上的嵌入数据集(parquet)转成 docs.npy / queries.npy。

两种数据集,字段已核对过(2026-10-23,读的是 HF 上的 parquet schema):

  openai   Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M
           列 _id / title / text / text-embedding-ada-002-1536-embedding(float32)
                                   / text-embedding-3-large-3072-embedding(float64)
           没有查询,也没有 GT。照 Extended RaBitQ 的做法从库里留出 1,000 条当查询
           (我们取最后 1,000 条,留出的不进库),GT 由 gt.py 自己算。
           --dim 3072 或 1536 选列;float64 转 float32。
  beir     CohereLabs/beir-embed-english-v3 的一个子集(nq、hotpotqa ……)
           corpus/*.parquet、queries/test.parquet:列 _id / text / emb(float32,1024 维)
           qrels/test.parquet:列 query_id / corpus_id / score
           输出 doc_ids.txt、query_ids.txt、qrels.tsv,供 ir_eval.py 算 nDCG@10。
           只保留 qrels 里出现过的查询(BEIR 的评测口径)。

docs.npy 以 memmap 逐文件写入,不整体放进内存(NQ 268 万 x 1024 x 4 B = 11 GB)。

用法:
  python hf_convert.py openai --src ~/hf/dbpedia3072 --dim 3072 --out ~/rbq_data/openai3072
  python hf_convert.py beir   --src ~/hf/beir/nq --out ~/rbq_data/beir_nq
"""
import argparse
import glob
import os

import numpy as np
import pyarrow.parquet as pq


def rows(files, column):
    n = 0
    for f in files:
        n += pq.ParquetFile(f).metadata.num_rows
    d = len(pq.ParquetFile(files[0]).read_row_group(0, columns=[column]).column(column)[0])
    return n, d


def stream(files, columns):
    for f in files:
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            yield pf.read_row_group(g, columns=columns)


def to_f32(col):
    a = np.asarray(col.combine_chunks().flatten())
    return a.astype(np.float32).reshape(len(col), -1)


def openai(a):
    col = "text-embedding-3-large-3072-embedding" if a.dim == 3072 else "text-embedding-ada-002-1536-embedding"
    files = sorted(glob.glob(os.path.join(a.src, "**", "*.parquet"), recursive=True))
    n, d = rows(files, col)
    nq = a.holdout
    print("openai-%d: %d 条,留出最后 %d 条当查询" % (d, n, nq))
    docs = np.lib.format.open_memmap(os.path.join(a.out, "docs.npy"), "w+", np.float32, (n - nq, d))
    qs = np.empty((nq, d), np.float32)
    pos = 0
    for t in stream(files, [col]):
        x = to_f32(t.column(col))
        e = pos + x.shape[0]
        nd = n - nq
        if pos < nd:
            docs[pos:min(e, nd)] = x[:min(e, nd) - pos]
        if e > nd:
            s = max(pos, nd)
            qs[s - nd:e - nd] = x[s - pos:]
        pos = e
        print("  %d / %d" % (pos, n), flush=True)
    docs.flush()
    np.save(os.path.join(a.out, "queries.npy"), qs)


def beir(a):
    files = sorted(glob.glob(os.path.join(a.src, "corpus", "*.parquet")))
    n, d = rows(files, "emb")
    print("beir corpus: %d 条, %d 维" % (n, d))
    docs = np.lib.format.open_memmap(os.path.join(a.out, "docs.npy"), "w+", np.float32, (n, d))
    pos = 0
    with open(os.path.join(a.out, "doc_ids.txt"), "w", encoding="utf-8") as fid:
        for t in stream(files, ["_id", "emb"]):
            x = to_f32(t.column("emb"))
            docs[pos:pos + x.shape[0]] = x
            fid.write("\n".join(t.column("_id").to_pylist()) + "\n")
            pos += x.shape[0]
            print("  %d / %d" % (pos, n), flush=True)
    docs.flush()
    qr = pq.read_table(os.path.join(a.src, "qrels", a.split + ".parquet")).to_pylist()
    judged = {r["query_id"] for r in qr}
    qt = pq.read_table(os.path.join(a.src, "queries", a.split + ".parquet"), columns=["_id", "emb"])
    qid = qt.column("_id").to_pylist()
    qv = to_f32(qt.column("emb"))
    keep = [i for i, q in enumerate(qid) if q in judged]
    np.save(os.path.join(a.out, "queries.npy"), qv[keep])
    with open(os.path.join(a.out, "query_ids.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(qid[i] for i in keep) + "\n")
    with open(os.path.join(a.out, "qrels.tsv"), "w", encoding="utf-8") as f:
        for r in qr:
            f.write("%s\t%s\t%d\n" % (r["query_id"], r["corpus_id"], r["score"]))
    print("queries: %d 个(有标注的)/ %d" % (len(keep), len(qid)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["openai", "beir"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dim", type=int, default=3072)
    ap.add_argument("--holdout", type=int, default=1000)
    ap.add_argument("--split", default="test", help="BEIR 的评测划分;msmarco 是 dev")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (openai if a.kind == "openai" else beir)(a)
    print("写出 %s/docs.npy, queries.npy" % a.out)


if __name__ == "__main__":
    main()
