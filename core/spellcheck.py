"""
spellcheck.py — Spelling correction using K-gram index and Levenshtein edit distance.
"""

from core.indexer import _generate_kgrams, tokenize, _get_stop_words, _stemmer

# Will be set by set_doc_frequencies() to enable frequency-based tie-breaking
_doc_freq: dict[str, int] = {}


# ---------------------------------------------------------------------------
# Levenshtein Edit Distance (pure Python, dynamic programming)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    m, n = len(s1), len(s2)
    # Use a single-row DP for space efficiency
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost # substitution
            )
        prev, curr = curr, prev

    return prev[n]


# ---------------------------------------------------------------------------
# K-gram candidate retrieval
# ---------------------------------------------------------------------------

def set_doc_frequencies(inverted_index: dict, vocabulary: set[str]):
    """Precompute document frequencies for vocabulary terms.

    This allows the spell checker to prefer more common words when
    multiple candidates have the same edit distance.
    """
    global _doc_freq
    from core.indexer import _stemmer
    _doc_freq = {}
    for term in vocabulary:
        stemmed = _stemmer.stem(term)
        if stemmed in inverted_index:
            _doc_freq[term] = len(inverted_index[stemmed])
        else:
            _doc_freq[term] = 0


def get_candidates(query_term: str,
                   kgram_index: dict[str, set[str]],
                   k: int = 2,
                   jaccard_threshold: float = 0.15) -> list[str]:
    """Retrieve candidate terms from the K-gram index via Jaccard overlap.

    Steps:
        1. Generate k-grams for the query term
        2. For each k-gram, collect vocabulary terms that share it
        3. Rank by Jaccard similarity of k-gram sets
        4. Return candidates above the threshold
    """
    query_kgrams = set(_generate_kgrams(query_term, k))
    if not query_kgrams:
        return []

    # Count how many k-grams each candidate shares with the query
    candidate_overlap: dict[str, int] = {}
    for kg in query_kgrams:
        for term in kgram_index.get(kg, []):
            candidate_overlap[term] = candidate_overlap.get(term, 0) + 1

    # Compute Jaccard and filter
    candidates = []
    min_len = max(1, len(query_term) - 2)
    for term, overlap in candidate_overlap.items():
        # Skip candidates that are too short (likely noise like "fl")
        if len(term) < min_len:
            continue
        term_kgrams = set(_generate_kgrams(term, k))
        union_size = len(query_kgrams | term_kgrams)
        jaccard = overlap / union_size if union_size > 0 else 0.0
        if jaccard >= jaccard_threshold:
            candidates.append(term)

    return candidates


# ---------------------------------------------------------------------------
# Correct a single term
# ---------------------------------------------------------------------------

def correct_term(term: str,
                 kgram_index: dict[str, set[str]],
                 vocabulary: set[str],
                 k: int = 2,
                 max_edit_distance: int = 2) -> str:
    """Suggest the best correction for a single term.

    Returns the original term if it is already in the vocabulary or if no
    good candidate is found.

    Tie-breaking strategy (when edit distance is equal):
        1. Prefer the candidate with higher document frequency (more common).
        2. If still tied, prefer the candidate closer in length to the query.
        3. If still tied, pick alphabetically for determinism.
    """
    if term in vocabulary:
        return term

    candidates = get_candidates(term, kgram_index, k)
    if not candidates:
        return term

    # Score each candidate
    scored = []
    for candidate in candidates:
        dist = levenshtein_distance(term, candidate)
        if dist <= max_edit_distance:
            df = _doc_freq.get(candidate, 0)
            len_diff = abs(len(candidate) - len(term))
            scored.append((dist, -df, len_diff, candidate))

    if not scored:
        return term

    # Sort by: edit distance (asc), -df (asc = higher df first), length diff (asc), alpha
    scored.sort()
    return scored[0][3]


# ---------------------------------------------------------------------------
# Correct a full query
# ---------------------------------------------------------------------------

def correct_query(query_text: str,
                  kgram_index: dict[str, set[str]],
                  vocabulary: set[str],
                  k: int = 2) -> tuple[str, bool]:
    """Correct all tokens in a query string.

    Parameters
    ----------
    query_text : raw user query string
    kgram_index : built by indexer.build_kgram_index
    vocabulary : set of all unstemmed terms in the corpus

    Returns
    -------
    corrected_query : str
        The query with misspelled tokens replaced.
    was_corrected : bool
        True if any token was changed.
    """
    sw = _get_stop_words()
    raw_tokens = tokenize(query_text)

    corrected_tokens = []
    was_corrected = False

    for token in raw_tokens:
        if token in sw:
            # Keep stopwords as-is (they won't be in vocabulary since we
            # removed them during indexing, but they shouldn't be "corrected")
            corrected_tokens.append(token)
            continue
        corrected = correct_term(token, kgram_index, vocabulary, k)
        if corrected != token:
            was_corrected = True
        corrected_tokens.append(corrected)

    corrected_query = " ".join(corrected_tokens)
    return corrected_query, was_corrected
