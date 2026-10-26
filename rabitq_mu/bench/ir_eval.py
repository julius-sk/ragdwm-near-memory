"""ir_eval.py —— 用 BEIR 的人工相关性标注(qrels)评测检索结果:nDCG@10 与 Recall@100 的 @k 版本。

eval.py 衡量的是「量化搜索找回了多少个精确最近邻」(ANN 召回);这里衡量的是「找回的文档
是不是人认为相关的」—— RAG 真正关心的指标。两者都报:精确最近邻本身的 nDCG(用 gt.py 的
GT 当作检索结果)是嵌入模型的上限,量化搜索的 nDCG 与它的差就是近似带来的损失。

nDCG@k 按 BEIR(pytrec_eval)的定义:gain = 相关度分数,折损 log2(rank+1),
理想排序用该查询全部标注文档。

用法:
  python ir_eval.py --conv ~/rbq_data/beir_nq --dump dev.bin --k 10
  python ir_eval.py --conv ~/rbq_data/beir_nq --gt nq.gt100.bin --k 10      # 精确搜索的上限
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval import read_dump  # noqa: E402
from gt import read_gt      # noqa: E402


def ndcg_at(ranked_docs, rel, k):
    dcg = sum(rel.get(d, 0) / np.log2(r + 2) for r, d in enumerate(ranked_docs[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", required=True, help="hf_convert.py beir 的输出目录")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dump")
    g.add_argument("--gt")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args()
    doc_ids = open(os.path.join(a.conv, "doc_ids.txt"), encoding="utf-8").read().split("\n")
    q_ids = open(os.path.join(a.conv, "query_ids.txt"), encoding="utf-8").read().split()
    qrels = defaultdict(dict)
    for line in open(os.path.join(a.conv, "qrels.tsv"), encoding="utf-8"):
        q, d, s = line.rstrip("\n").split("\t")
        if int(s) > 0:
            qrels[q][d] = int(s)
    if a.dump:
        _, ids, _, _ = read_dump(a.dump, a.k)
    else:
        ids, _ = read_gt(a.gt)
        ids = ids[:, :a.k]
    nq = ids.shape[0]
    nd, rc = [], []
    for i in range(nq):
        ranked = [doc_ids[j] for j in ids[i] if 0 <= j < len(doc_ids)]
        rel = qrels[q_ids[i]]
        nd.append(ndcg_at(ranked, rel, a.k))
        rc.append(len(set(ranked[:a.k]) & set(rel)) / max(1, len(rel)))
    out = {"src": a.dump or a.gt, "nq": nq, "ndcg@%d" % a.k: float(np.mean(nd)),
           "recall@%d_rel" % a.k: float(np.mean(rc))}
    print("%s: %d 个查询  nDCG@%d = %.4f   相关文档召回@%d = %.4f"
          % (out["src"], nq, a.k, out["ndcg@%d" % a.k], a.k, out["recall@%d_rel" % a.k]))
    if a.json:
        with open(a.json, "a") as f:
            f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
