"""kernel 算法的 Python 模型 —— 与 mu_hamming.cpp 逐行对应。

用途:本地(无 XCENA、无 C++ 编译器)验证计数排序 top-k 与溢出重试的正确性。
模型改了,C++ 也要改;两边逻辑必须保持一致。
"""
import numpy as np

from verify_host import BITS, NBUCKETS, WORDS, gen_sigs, hamming_all, topk_ids


def _task_range(num_sigs, task_idx, task_count):
    """与 kernel 的 myRange 逐行对应:均分,最后一个吃余数。"""
    per_task = (num_sigs + task_count - 1) // task_count
    begin = task_idx * per_task
    end = min(begin + per_task, num_sigs)
    return min(begin, num_sigs), end


def scan_hist(sigs, query, task_count, coarse_thresh, cand_cap):
    """模拟 hamming_scan_hist:每 task 一份 257 桶直方图 + 候选分片。

    候选交错存放 (id, dist):cand[t, 2j]=id, cand[t, 2j+1]=dist。
    存距离是正确性的必要条件 —— 见 merge_topk 的过滤。
    """
    n = len(sigs)
    dists = hamming_all(sigs, query)
    hist = np.zeros((task_count, NBUCKETS), dtype=np.uint32)
    cand = np.zeros((task_count, cand_cap * 2), dtype=np.uint32)   # 交错 (id, dist)
    meta = np.zeros((task_count, 2), dtype=np.uint32)   # [candCount, overflow]

    for t in range(task_count):
        begin, end = _task_range(n, t, task_count)
        n_cand = 0
        overflow = 0
        for i in range(begin, end):
            d = int(dists[i])
            hist[t, d] += 1                      # 直方图统计全部,不受 cap 影响
            if d <= coarse_thresh:
                if n_cand < cand_cap:
                    cand[t, 2 * n_cand] = i
                    cand[t, 2 * n_cand + 1] = d  # ★ 必须连距离一起存
                    n_cand += 1
                else:
                    overflow = 1                 # 记录但不中断
        meta[t, 0] = n_cand
        meta[t, 1] = overflow
    return hist, cand, meta


def merge_topk(hist, cand, meta, top_k):
    """模拟 hamming_merge_topk:归并直方图定精确阈值,按阈值过滤候选。"""
    task_count = hist.shape[0]
    total = hist.sum(axis=0, dtype=np.uint64)

    exact_thresh = BITS
    cum = 0
    for b in range(NBUCKETS):
        cum += int(total[b])
        if cum >= top_k:
            exact_thresh = b
            break

    any_overflow = int(meta[:, 1].any())

    # ★ 只收 dist <= exact_thresh 的候选。没有这一步,粗阈值偏大时会返回
    #   宽带里的任意 id(已复现的静默错误:返回距离 145 而真实第k名是 113)。
    ids = []
    for t in range(task_count):
        if len(ids) >= top_k:
            break
        cnt = int(meta[t, 0])
        for j in range(cnt):
            if len(ids) >= top_k:
                break
            if int(cand[t, 2 * j + 1]) <= exact_thresh:
                ids.append(int(cand[t, 2 * j]))

    collected = len(ids)
    return np.array(ids, dtype=np.uint32), exact_thresh, any_overflow, collected


def run_with_retry(sigs, query, task_count, top_k, cand_cap, thresh0, max_attempts=5):
    """模拟 host 端的重试循环。

    因为 merge 已按 exact_thresh 过滤,收集到的 id 必然满足 d <= 第k名距离,
    所以**溢出不会产生错误结果**,只可能导致收集不足。重试条件因此只看数量:
      collected >= top_k          -> 完成(即便 overflow=1)
      collected <  top_k          -> 令 thresh = exact_thresh 重跑
                                     (过紧则放宽、过松则收紧,两个方向都收敛)
      已在 exact_thresh 仍不足    -> cand_cap 太小,调用方需增大它
    """
    thresh = thresh0
    attempts = 0
    ids = np.array([], dtype=np.uint32)
    for _ in range(max_attempts):
        hist, cand, meta = scan_hist(sigs, query, task_count, thresh, cand_cap)
        ids, exact_thresh, any_overflow, collected = merge_topk(hist, cand, meta, top_k)
        if collected >= top_k:
            break
        if thresh == exact_thresh:
            break            # 已在精确阈值仍不足 -> cand_cap 受限,重试无用
        thresh = exact_thresh
        attempts += 1
    return ids, attempts


def _fuzz():
    rng = np.random.default_rng(7)
    for trial in range(200):
        n = int(rng.integers(1, 3000))
        tc = int(rng.integers(1, 65))
        k = int(rng.integers(1, min(n, 200) + 1))
        cap = int(rng.integers(1, 300))
        thresh = int(rng.integers(0, BITS + 1))

        sigs = gen_sigs(n, seed=trial)
        query = gen_sigs(1, seed=trial + 9999)[0]
        ref = hamming_all(sigs, query)

        ids, attempts = run_with_retry(sigs, query, tc, k, cap, thresh)

        # 不变式 1:凑够 k 个(除非 cap*tc 物理上装不下 k 个)
        if cap * tc >= k:
            assert len(ids) == k, f"trial {trial}: 只收到 {len(ids)}/{k}"
            # 不变式 2:每个返回 id 的距离必须 <= 第 k 名的距离(平局允许换人)
            kth = np.sort(ref)[k - 1]
            assert ref[ids].max() <= kth, (
                f"trial {trial}: 返回距离 {ref[ids].max()} > 第k名 {kth}")
            # 不变式 3:无重复 id
            assert len(set(ids.tolist())) == len(ids), f"trial {trial}: 有重复 id"
        # 不变式 4:重试次数有界
        assert attempts <= 5, f"trial {trial}: 重试 {attempts} 次未收敛"

    # 定向用例:阈值极紧(收集不足)与极松(必然溢出)都要能自愈
    sigs = gen_sigs(2000, seed=1)
    query = gen_sigs(1, seed=2)[0]
    ref = hamming_all(sigs, query)
    for thresh0 in (0, BITS):
        ids, attempts = run_with_retry(sigs, query, 4, 50, 60, thresh0)
        kth = np.sort(ref)[49]
        assert len(ids) == 50 and ref[ids].max() <= kth, f"thresh0={thresh0} 未自愈"
        assert attempts >= 1, f"thresh0={thresh0} 应触发至少一次重试"

    # 直方图必须统计全部签名,不受 cand_cap 影响(溢出自愈的前提)
    hist, _cand, _meta = scan_hist(sigs, query, 4, coarse_thresh=BITS, cand_cap=1)
    assert hist.sum() == len(sigs), "直方图丢了签名 —— 溢出自愈的前提被破坏"

    # ---- 回归用例:粗阈值远大于精确阈值时的静默错误 ----
    # 初版设计(候选只存 id、merge 不按 exact_thresh 过滤)在此处会
    # "成功门通过但结果错":返回距离 145,而真实第 k 名距离是 113。
    # 这个用例锁死该 bug,任何回退到"不过滤"的改动都会在这里响亮失败。
    sigs = gen_sigs(2000, seed=0)
    query = gen_sigs(1, seed=9999)[0]
    ref = hamming_all(sigs, query)
    k = 50
    kth = int(np.sort(ref)[k - 1])
    for coarse in (BITS, 148, 130, kth):          # 从极松到刚好
        hist, cand, meta = scan_hist(sigs, query, 4, coarse, cand_cap=100000)
        ids, exact_thresh, _ov, collected = merge_topk(hist, cand, meta, k)
        assert exact_thresh == kth, f"coarse={coarse}: 精确阈值 {exact_thresh} != 第k名 {kth}"
        assert collected == k, f"coarse={coarse}: 只收集到 {collected}/{k}"
        assert ref[ids].max() <= kth, (
            f"coarse={coarse}: 返回距离 {ref[ids].max()} > 第k名 {kth} —— "
            "候选未按 exact_thresh 过滤(静默错误回归)")

    print("model fuzz: ALL PASS (200 随机 + 定向用例 + 松阈值回归)")


if __name__ == "__main__":
    _fuzz()
