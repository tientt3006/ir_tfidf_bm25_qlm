"""
models.py — Ranking models: TF-IDF, BM25, QLM-JM, QLM-Dirichlet.

All models operate on the inverted index built by indexer.py.
"""

import math
from collections import defaultdict


# ---------------------------------------------------------------------------
# TF-IDF (lnc.ltc variant — log-tf, idf, cosine normalisation)
# ---------------------------------------------------------------------------

def tfidf_rank(query_tokens: list[str],
               inverted_index: dict,
               tf_raw: dict,
               doc_lengths: dict,
               N: int,
               top_k: int = 10) -> list[tuple[int, float]]:
    """Rank documents using TF-IDF with cosine normalisation.

    TF  = 1 + log10(raw_tf)  if raw_tf > 0
    IDF = log10(N / df)
    Score = sum of (tf * idf) for each query term, normalised by doc vector length.
    """
    scores: dict[int, float] = defaultdict(float)
    doc_norm: dict[int, float] = defaultdict(float)

    for term in query_tokens:
        if term not in inverted_index:
            continue
        posting = inverted_index[term]
        df = len(posting)
        idf = math.log10(N / df) if df > 0 else 0.0

        for doc_id, positions in posting.items():
            tf = 1.0 + math.log10(len(positions))
            weight = tf * idf
            scores[doc_id] += weight

    # Normalise by document vector length (precompute per-document)
    # We approximate with sqrt(doc_length) for efficiency, which is a
    # standard pivot normalisation proxy.
    for doc_id in scores:
        dl = doc_lengths.get(doc_id, 1)
        if dl > 0:
            scores[doc_id] /= math.sqrt(dl)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Okapi BM25
# ---------------------------------------------------------------------------

def bm25_rank(query_tokens: list[str],
              inverted_index: dict,
              tf_raw: dict,
              doc_lengths: dict,
              avg_dl: float,
              N: int,
              k1: float = 1.2,
              b: float = 0.75,
              top_k: int = 10) -> list[tuple[int, float]]:
    """Rank documents using BM25.

    score(D, Q) = Σ IDF(qi) · [ f(qi,D)·(k1+1) ] / [ f(qi,D) + k1·(1 - b + b·|D|/avgdl) ]
    IDF(qi) = log( (N - df + 0.5) / (df + 0.5) + 1 )
    """
    scores: dict[int, float] = defaultdict(float)

    for term in query_tokens:
        if term not in inverted_index:
            continue
        posting = inverted_index[term]
        df = len(posting)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id in posting:
            tf = tf_raw[term][doc_id]
            dl = doc_lengths[doc_id]
            numerator = tf * (k1 + 1.0)
            denominator = tf + k1 * (1.0 - b + b * dl / avg_dl)
            scores[doc_id] += idf * numerator / denominator

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Query Likelihood Model — Jelinek-Mercer Smoothing
# ---------------------------------------------------------------------------

def _build_collection_stats(tf_raw: dict, doc_lengths: dict):
    """Precompute collection-level statistics for QLM models.

    Returns
    -------
    cf_cache : dict[str, float]
        P(t|C) for each term
    total_tokens : int
        Total number of tokens in the collection
    """
    total_tokens = sum(doc_lengths.values())
    cf_cache = {}
    for term, postings in tf_raw.items():
        cf = sum(postings.values())
        cf_cache[term] = cf / total_tokens if total_tokens > 0 else 0.0
    return cf_cache, total_tokens


# Module-level cache (populated lazily)
_cf_cache: dict[str, float] | None = None
_total_tokens: int = 0


def _get_collection_stats(tf_raw: dict, doc_lengths: dict):
    """Get or build the collection stats cache."""
    global _cf_cache, _total_tokens
    if _cf_cache is None:
        _cf_cache, _total_tokens = _build_collection_stats(tf_raw, doc_lengths)
    return _cf_cache, _total_tokens


def qlm_jm_rank(query_tokens: list[str],
                inverted_index: dict,
                tf_raw: dict,
                doc_lengths: dict,
                N: int,
                lam: float = 0.4,
                top_k: int = 10) -> list[tuple[int, float]]:
    """Rank documents using Query Likelihood Model with Jelinek-Mercer smoothing.

    P(Q|D) = prod [ lam * P(t|D) + (1 - lam) * P(t|C) ]
    Score  = sum log[ lam * (tf_d / |D|) + (1 - lam) * P(t|C) ]
    """
    cf_cache, total_tokens = _get_collection_stats(tf_raw, doc_lengths)
    scores: dict[int, float] = {}

    # Collect candidate documents (any doc containing at least one query term)
    candidate_docs: set[int] = set()
    for term in query_tokens:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term].keys())

    for doc_id in candidate_docs:
        dl = doc_lengths[doc_id]
        score = 0.0
        for term in query_tokens:
            tf_d = tf_raw.get(term, {}).get(doc_id, 0)
            p_td = tf_d / dl if dl > 0 else 0.0
            p_tc = cf_cache.get(term, 0.0)
            smoothed = lam * p_td + (1.0 - lam) * p_tc
            if smoothed > 0:
                score += math.log(smoothed)
            else:
                score += -1e9  # very small log probability
        scores[doc_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Query Likelihood Model — Dirichlet Prior Smoothing
# ---------------------------------------------------------------------------

def qlm_dirichlet_rank(query_tokens: list[str],
                       inverted_index: dict,
                       tf_raw: dict,
                       doc_lengths: dict,
                       N: int,
                       mu: float = 1500.0,
                       top_k: int = 10) -> list[tuple[int, float]]:
    """Rank documents using Query Likelihood Model with Dirichlet Prior smoothing.

    P(t|D) = (tf_d + mu * P(t|C)) / (|D| + mu)
    Score  = sum log P(t|D)
    """
    cf_cache, total_tokens = _get_collection_stats(tf_raw, doc_lengths)
    scores: dict[int, float] = {}

    candidate_docs: set[int] = set()
    for term in query_tokens:
        if term in inverted_index:
            candidate_docs.update(inverted_index[term].keys())

    for doc_id in candidate_docs:
        dl = doc_lengths[doc_id]
        score = 0.0
        for term in query_tokens:
            tf_d = tf_raw.get(term, {}).get(doc_id, 0)
            p_tc = cf_cache.get(term, 0.0)
            smoothed = (tf_d + mu * p_tc) / (dl + mu)
            if smoothed > 0:
                score += math.log(smoothed)
            else:
                score += -1e9
        scores[doc_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Unified ranking interface
# ---------------------------------------------------------------------------

MODELS = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "QLM (JM)": "qlm_jm",
    "QLM (Dirichlet)": "qlm_dirichlet",
}


def rank(model_name: str,
         query_tokens: list[str],
         inverted_index: dict,
         tf_raw: dict,
         doc_lengths: dict,
         avg_dl: float,
         N: int,
         top_k: int = 10,
         **params) -> list[tuple[int, float]]:
    """Unified ranking dispatcher.

    Parameters
    ----------
    model_name : one of MODELS keys
    params : model-specific keyword arguments (k1, b, lam, mu)
    """
    key = MODELS.get(model_name, model_name)
    if key == "tfidf":
        return tfidf_rank(query_tokens, inverted_index, tf_raw, doc_lengths, N, top_k)
    elif key == "bm25":
        k1 = params.get("k1", 1.2)
        b = params.get("b", 0.75)
        return bm25_rank(query_tokens, inverted_index, tf_raw, doc_lengths, avg_dl, N, k1, b, top_k)
    elif key == "qlm_jm":
        lam = params.get("lam", 0.4)
        return qlm_jm_rank(query_tokens, inverted_index, tf_raw, doc_lengths, N, lam, top_k)
    elif key == "qlm_dirichlet":
        mu = params.get("mu", 1500.0)
        return qlm_dirichlet_rank(query_tokens, inverted_index, tf_raw, doc_lengths, N, mu, top_k)
    else:
        raise ValueError(f"Unknown model: {model_name}")
