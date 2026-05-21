"""
indexer.py — Text preprocessing, Inverted Index, and K-gram Index construction.

Uses the Cranfield collection loaded via ir_datasets.
"""

import os
import re
import json
import math
from collections import defaultdict

import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
_NLTK_DATA_READY = False

def _ensure_nltk_data():
    global _NLTK_DATA_READY
    if _NLTK_DATA_READY:
        return
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_DATA_READY = True

# ---------------------------------------------------------------------------
# Globals initialised lazily
# ---------------------------------------------------------------------------
_stemmer = PorterStemmer()
_stop_words = None

def _get_stop_words():
    global _stop_words
    if _stop_words is None:
        _ensure_nltk_data()
        _stop_words = set(stopwords.words("english"))
    return _stop_words

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-word characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def remove_stopwords(tokens: list[str]) -> list[str]:
    sw = _get_stop_words()
    return [t for t in tokens if t not in sw]


def stem(tokens: list[str]) -> list[str]:
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """Full preprocessing pipeline: tokenize → remove stopwords → stem."""
    return stem(remove_stopwords(tokenize(text)))


def preprocess_keep_original(text: str) -> tuple[list[str], list[str]]:
    """Return (stemmed_tokens, original_tokens_after_stopword_removal).

    Useful for spell-check: we need the *unstemmed* vocabulary to suggest
    corrections that a user can read.
    """
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    stemmed = stem(tokens)
    return stemmed, tokens

# ---------------------------------------------------------------------------
# Cranfield data loading
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cranfield")


def load_cranfield():
    """Load the Cranfield collection.

    Returns
    -------
    documents : dict[int, dict]
        {doc_id: {"title": str, "body": str, "author": str, "bib": str}}
    queries : dict[int, str]
        {query_id: query_text}
    qrels : dict[int, set[int]]
        {query_id: set_of_relevant_doc_ids}
    """
    cache_file = os.path.join(_DATA_DIR, "cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        documents = {int(k): v for k, v in data["documents"].items()}
        queries = {int(k): v for k, v in data["queries"].items()}
        qrels = {int(k): set(v) for k, v in data["qrels"].items()}
        return documents, queries, qrels

    dataset = ir_datasets.load("cranfield")

    # Documents
    documents = {}
    for doc in dataset.docs_iter():
        documents[int(doc.doc_id)] = {
            "title": doc.title,
            "body": doc.text,
            "author": doc.author,
            "bib": doc.bib,
        }

    # Queries
    queries = {}
    for query in dataset.queries_iter():
        queries[int(query.query_id)] = query.text

    # Qrels — Cranfield uses *negative* relevance scores where lower is better
    # (1 = most relevant, 5 = least).  We keep all as relevant for standard
    # evaluation (common practice).  Filter if desired.
    qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        qrels[int(qrel.query_id)].add(int(qrel.doc_id))
    qrels = dict(qrels)

    # Cache to disk
    os.makedirs(_DATA_DIR, exist_ok=True)
    cache = {
        "documents": {str(k): v for k, v in documents.items()},
        "queries": {str(k): v for k, v in queries.items()},
        "qrels": {str(k): list(v) for k, v in qrels.items()},
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f)
        
    # Also dump readable individual files for developers to inspect
    with open(os.path.join(_DATA_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(cache["documents"], f, indent=2)
    with open(os.path.join(_DATA_DIR, "queries.json"), "w", encoding="utf-8") as f:
        json.dump(cache["queries"], f, indent=2)
    with open(os.path.join(_DATA_DIR, "qrels.json"), "w", encoding="utf-8") as f:
        json.dump(cache["qrels"], f, indent=2)
        json.dump(cache, f, ensure_ascii=False)

    return documents, queries, qrels

# ---------------------------------------------------------------------------
# Inverted Index
# ---------------------------------------------------------------------------

def build_inverted_index(documents: dict[int, dict]):
    """Build an inverted index from the Cranfield documents.

    Parameters
    ----------
    documents : dict returned by load_cranfield()

    Returns
    -------
    inverted_index : dict[str, dict[int, list[int]]]
        term → {doc_id → [positions]}
    doc_lengths : dict[int, int]
        doc_id → number of tokens in document
    avg_dl : float
        Average document length
    N : int
        Total number of documents
    tf_raw : dict[str, dict[int, int]]
        term → {doc_id → raw term frequency}
    vocabulary : set[str]
        Set of all *unstemmed* terms (after stopword removal) in the corpus,
        for use by the spell-checker.
    """
    inverted_index: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    doc_lengths: dict[int, int] = {}
    tf_raw: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    vocabulary: set[str] = set()

    for doc_id, doc in documents.items():
        text = doc["title"] + " " + doc["body"]
        stemmed_tokens, original_tokens = preprocess_keep_original(text)
        vocabulary.update(original_tokens)
        doc_lengths[doc_id] = len(stemmed_tokens)
        for pos, token in enumerate(stemmed_tokens):
            inverted_index[token][doc_id].append(pos)
            tf_raw[token][doc_id] += 1

    N = len(documents)
    avg_dl = sum(doc_lengths.values()) / N if N > 0 else 0.0

    # Convert defaultdicts to plain dicts for cleanliness
    inverted_index = {k: dict(v) for k, v in inverted_index.items()}
    tf_raw = {k: dict(v) for k, v in tf_raw.items()}

    return inverted_index, doc_lengths, avg_dl, N, tf_raw, vocabulary

# ---------------------------------------------------------------------------
# K-gram Index (for spell-checking)
# ---------------------------------------------------------------------------

def _generate_kgrams(term: str, k: int = 2) -> list[str]:
    """Generate k-grams for a term with '$' padding."""
    padded = "$" + term + "$"
    return [padded[i:i + k] for i in range(len(padded) - k + 1)]


def build_kgram_index(vocabulary, k: int = 2):
    """Build a k-gram index from a vocabulary set.

    Returns
    -------
    kgram_index : dict[str, set[str]]
        k-gram → set of vocabulary terms containing that k-gram
    """
    kgram_index: dict[str, set[str]] = defaultdict(set)
    for term in vocabulary:
        for kg in _generate_kgrams(term, k):
            kgram_index[kg].add(term)
    return dict(kgram_index)
