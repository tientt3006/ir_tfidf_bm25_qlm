"""
metrics.py — Evaluation metrics implemented from scratch (no scikit-learn).

Functions operate on lists of document IDs (retrieved and relevant).
"""


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Precision at rank position k.

    P@K = |{relevant docs in top-k}| / k
    """
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Recall at rank position k.

    R@K = |{relevant docs in top-k}| / |relevant|
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant)
    return relevant_in_top_k / len(relevant)


def f1_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """F1-Score at rank position k.

    F1@K = 2 * P@K * R@K / (P@K + R@K)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def average_precision(retrieved: list[int], relevant: set[int]) -> float:
    """Average Precision (AP) for a single query.

    AP = (1 / |relevant|) * Σ_{k=1}^{n} P@k * rel(k)
    where rel(k) = 1 if retrieved[k-1] is relevant, else 0.
    """
    if not relevant:
        return 0.0
    num_relevant_seen = 0
    sum_precision = 0.0
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            num_relevant_seen += 1
            sum_precision += num_relevant_seen / k
    return sum_precision / len(relevant)


def mean_average_precision(results: dict[int, list[int]],
                           qrels: dict[int, set[int]]) -> float:
    """Mean Average Precision (MAP) across all queries.

    Parameters
    ----------
    results : {query_id: [doc_ids in ranked order]}
    qrels   : {query_id: set of relevant doc_ids}
    """
    if not results:
        return 0.0
    ap_sum = 0.0
    count = 0
    for qid, retrieved in results.items():
        if qid in qrels:
            ap_sum += average_precision(retrieved, qrels[qid])
            count += 1
    return ap_sum / count if count > 0 else 0.0


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    """Reciprocal Rank (RR) for a single query.

    RR = 1 / rank_of_first_relevant_doc
    """
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / k
    return 0.0


def mean_reciprocal_rank(results: dict[int, list[int]],
                         qrels: dict[int, set[int]]) -> float:
    """Mean Reciprocal Rank (MRR) across all queries.

    Parameters
    ----------
    results : {query_id: [doc_ids in ranked order]}
    qrels   : {query_id: set of relevant doc_ids}
    """
    if not results:
        return 0.0
    rr_sum = 0.0
    count = 0
    for qid, retrieved in results.items():
        if qid in qrels:
            rr_sum += reciprocal_rank(retrieved, qrels[qid])
            count += 1
    return rr_sum / count if count > 0 else 0.0


def evaluate_all(results: dict[int, list[int]],
                 qrels: dict[int, set[int]],
                 k: int = 10) -> dict[str, float]:
    """Compute all metrics at once.

    Returns dict with keys: P@K, R@K, F1@K, MAP, MRR.
    P@K, R@K, F1@K are macro-averaged across queries.
    """
    p_sum = r_sum = f1_sum = 0.0
    count = 0
    for qid, retrieved in results.items():
        if qid in qrels:
            rel = qrels[qid]
            p_sum += precision_at_k(retrieved, rel, k)
            r_sum += recall_at_k(retrieved, rel, k)
            f1_sum += f1_at_k(retrieved, rel, k)
            count += 1

    n = count if count > 0 else 1
    return {
        f"P@{k}": p_sum / n,
        f"R@{k}": r_sum / n,
        f"F1@{k}": f1_sum / n,
        "MAP": mean_average_precision(results, qrels),
        "MRR": mean_reciprocal_rank(results, qrels),
    }
