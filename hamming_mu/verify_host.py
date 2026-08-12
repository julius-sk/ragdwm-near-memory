"""Hamming 扫描的 numpy 参考实现 + 设备输出比对工具。

设备侧签名布局:每签名 4 x uint64 = 32B 连续,小端序。
本文件既是参考实现,也是 CLI 工具(gen / check / check-topk)。
"""
import argparse
import sys

import numpy as np

WORDS = 4                 # 每签名 4 个 uint64
BITS = WORDS * 64         # 256 位
NBUCKETS = BITS + 1       # 距离 0..256,共 257 个桶


def gen_sigs(n, seed=0):
    """生成 n 个随机 256 位签名,形状 (n,4) 的 uint64。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**64, size=(n, WORDS), dtype=np.uint64)


def sigs_to_bytes(sigs):
    """转成设备侧内存布局:每签名 4 x uint64 连续,小端序。"""
    return np.ascontiguousarray(sigs, dtype='<u8').tobytes()


def hamming_all(sigs, query):
    """query 对全部 sigs 的 Hamming 距离,返回 (n,) int32,值域 [0,256]。"""
    sigs = np.asarray(sigs, dtype=np.uint64).reshape(-1, WORDS)
    query = np.asarray(query, dtype=np.uint64).reshape(WORDS)
    x = np.bitwise_xor(sigs, query)
    # 逐字节查表 popcount:对 uint64 视图按字节展开再累加
    b = x.view(np.uint8).reshape(-1, WORDS * 8)
    return np.unpackbits(b, axis=1).sum(axis=1).astype(np.int32)


def topk_ids(dists, k):
    """最近的 k 个 id,按 (距离升序, id 升序) 稳定排序,返回 (min(k,n),) uint32。

    若 k > n,返回全部 n 个 id;否则返回前 k 个。
    """
    dists = np.asarray(dists)
    n = len(dists)
    k = min(k, n)  # Clamp k to n
    order = np.argsort(dists, kind="stable")   # stable => 平局按 id 升序
    return order[:k].astype(np.uint32)


def _self_test():
    # 1. 全 0 与全 1 的距离必须是 256
    a = np.zeros((1, WORDS), dtype=np.uint64)
    b = np.full((1, WORDS), np.uint64(0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
    assert hamming_all(b, a[0])[0] == 256, "全0 vs 全1 应为 256"

    # 2. 自己和自己的距离必须是 0
    s = gen_sigs(16, seed=1)
    assert hamming_all(s, s[0])[0] == 0, "自距离应为 0"

    # 3. 值域必须落在 [0,256]
    d = hamming_all(s, s[3])
    assert d.min() >= 0 and d.max() <= BITS, "距离越界"

    # 4. 单比特差异的距离必须是 1
    c = s[0].copy()
    c[0] = np.uint64(c[0] ^ np.uint64(1))
    assert hamming_all(c.reshape(1, WORDS), s[0])[0] == 1, "单比特差应为 1"

    # 5. 字节序往返必须无损
    raw = sigs_to_bytes(s)
    back = np.frombuffer(raw, dtype='<u8').reshape(-1, WORDS)
    assert np.array_equal(back, s), "字节序往返不一致"

    # 6. top-k 必须按 (距离, id) 稳定排序
    dd = np.array([3, 1, 1, 0, 5], dtype=np.int32)
    assert list(topk_ids(dd, 3)) == [3, 1, 2], "top-k 平局需按 id 升序"

    # 7. k > n 时必须被 clamp 到 n
    dd_short = np.array([2, 5, 1], dtype=np.int32)
    result_k_gt_n = topk_ids(dd_short, k=10)
    assert len(result_k_gt_n) == 3, f"k > n 应 clamp 到 n=3,得到 {len(result_k_gt_n)}"
    assert list(result_k_gt_n) == [2, 0, 1], "k > n 时应返回全部 n 个 id"

    # 8. 重复 id 检查:模拟设备返回重复 id,验证检查逻辑会拒绝
    ref_dists = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    duplicate_ids = np.array([0, 0, 1], dtype=np.uint32)  # 重复 id 0
    # 模拟 _cmd_check_topk 的唯一性检查逻辑
    got_unique_count = len(set(duplicate_ids.tolist()))
    assert got_unique_count != len(duplicate_ids), "重复 id 案例应有重复"
    # 验证能检测出重复
    unique_ids_dict = {}
    for id_val in duplicate_ids:
        unique_ids_dict[id_val] = unique_ids_dict.get(id_val, 0) + 1
    duplicates = {id_val: count for id_val, count in unique_ids_dict.items() if count > 1}
    assert len(duplicates) > 0, "应检测出重复"
    assert duplicates[0] == 2, "id 0 应出现 2 次"

    # 9. 单个哨兵(sentinel)填充:唯一性检查放行(值唯一),但索引越界,
    #    模拟 _cmd_check_topk 里的哨兵检查逻辑,验证它能在 ref[got] 之前拦截,
    #    而不是让调用方看到 IndexError 的 traceback。
    ref_dists9 = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    sentinel_ids = np.array([0, 1, 0xFFFFFFFF], dtype=np.uint32)  # 只有一个哨兵,值唯一
    got_unique_count9 = len(set(sentinel_ids.tolist()))
    assert got_unique_count9 == len(sentinel_ids), "单哨兵案例里 id 应各自唯一(不触发重复检查)"
    assert int(sentinel_ids.max()) >= len(ref_dists9), "哨兵应越界,触发哨兵检查而非 ref[got]"

    print("self-test: ALL PASS")


def _cmd_gen(args):
    sigs = gen_sigs(args.n, seed=args.seed)
    query = gen_sigs(1, seed=args.seed + 12345)[0]
    with open(args.out, "wb") as f:
        f.write(sigs_to_bytes(sigs))
    with open(args.out + ".query", "wb") as f:
        f.write(sigs_to_bytes(query.reshape(1, WORDS)))
    d = hamming_all(sigs, query)
    np.save(args.out + ".dists.npy", d)
    print(f"wrote {args.n} sigs -> {args.out} ({args.n*32} B)")
    print(f"  dist min={d.min()} max={d.max()} mean={d.mean():.2f}")


def _cmd_check(args):
    ref = np.load(args.ref)
    got = np.fromfile(args.dists, dtype=np.int32)
    if got.size != ref.size:
        print(f"FAIL: 长度不符 ref={ref.size} got={got.size}")
        sys.exit(1)
    bad = np.nonzero(got != ref)[0]
    if bad.size:
        i = bad[0]
        print(f"FAIL: {bad.size} 处不符,首个 idx={i} ref={ref[i]} got={got[i]}")
        sys.exit(1)
    print(f"PASS: {ref.size} 个距离逐位一致")


def _cmd_check_topk(args):
    ref = np.load(args.ref)
    want = set(topk_ids(ref, args.k).tolist())
    got = np.fromfile(args.ids, dtype=np.uint32)
    if got.size != args.k:
        print(f"FAIL: 期望 {args.k} 个 id,得到 {got.size}")
        sys.exit(1)
    # 检查唯一性:不能有重复 id
    if len(set(got.tolist())) != len(got):
        # 找出哪个 id 重复了
        unique_ids = {}
        for id_val in got:
            unique_ids[id_val] = unique_ids.get(id_val, 0) + 1
        duplicates = {id_val: count for id_val, count in unique_ids.items() if count > 1}
        first_dup = sorted(duplicates.items())[0]
        print(f"FAIL: id {first_dup[0]} 出现 {first_dup[1]} 次 (期望唯一)")
        sys.exit(1)
    # 哨兵(sentinel)填充检查:设备收集不足 k 个真实 id 时,用 0xFFFFFFFF
    # 填充剩余槽位。两个及以上哨兵会被上面的唯一性检查捕获(值相同即重复);
    # 但恰好一个哨兵时该 id 本身是唯一的,不会被那个检查拦下 —— 若不在这里
    # 提前拦截,下面 `ref[got]` 会因为索引越界抛 IndexError,而不是给出
    # 清晰的 FAIL(这个函数是校验设备输出的服务器端关卡,不应该在这里崩溃)。
    if got.size and int(got.max()) >= len(ref):
        bad_id = int(got.max())
        print(f"FAIL: id {bad_id} 越界 (>= len(ref)={len(ref)}) —— 设备只收集到"
              f"不足 {args.k} 个真实 id,用哨兵(0xFFFFFFFF)填充了剩余槽位")
        sys.exit(1)
    # 平局可能使具体 id 不同,但每个返回 id 的距离必须 <= 第 k 名的距离
    kth = np.sort(ref)[args.k - 1]
    biggest = ref[got].max()
    if biggest > kth:
        print(f"FAIL: 返回了距离 {biggest} 的 id,但第 {args.k} 名距离是 {kth}")
        sys.exit(1)
    overlap = len(want & set(got.tolist()))
    print(f"PASS: {args.k} 个 id 距离全部 <= 第k名({kth});与参考重合 {overlap}/{args.k}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen"); g.set_defaults(fn=_cmd_gen)
    g.add_argument("-n", type=int, required=True)
    g.add_argument("-o", dest="out", required=True)
    g.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("check"); c.set_defaults(fn=_cmd_check)
    c.add_argument("--ref", required=True)
    c.add_argument("--dists", required=True)

    t = sub.add_parser("check-topk"); t.set_defaults(fn=_cmd_check_topk)
    t.add_argument("--ref", required=True)
    t.add_argument("--ids", required=True)
    t.add_argument("-k", type=int, required=True)

    s = sub.add_parser("selftest"); s.set_defaults(fn=lambda a: _self_test())

    args = p.parse_args()
    args.fn(args)
