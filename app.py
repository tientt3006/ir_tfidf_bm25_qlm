"""
app.py — Streamlit UI for the Information Retrieval system.

Run with: streamlit run app.py
"""

import os
import sys
import json
import time

import streamlit as st

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from core.indexer import (
    load_cranfield, build_inverted_index, build_kgram_index, preprocess
)
from core.models import rank, MODELS
from core.spellcheck import correct_query, set_doc_frequencies

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Cranfield IR System",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load best params from tuning (if available)
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BEST_PARAMS_FILE = os.path.join(RESULTS_DIR, "best_params.json")

DEFAULT_PARAMS = {
    "BM25": {"k1": 1.2, "b": 0.75},
    "QLM (JM)": {"lam": 0.4},
    "QLM (Dirichlet)": {"mu": 1500.0},
}

if os.path.exists(BEST_PARAMS_FILE):
    with open(BEST_PARAMS_FILE, "r") as f:
        best_params = json.load(f)
    # Merge with defaults (in case some keys are missing)
    for k in DEFAULT_PARAMS:
        if k not in best_params:
            best_params[k] = DEFAULT_PARAMS[k]
else:
    best_params = DEFAULT_PARAMS

# ---------------------------------------------------------------------------
# Cache data loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Cranfield dataset...")
def load_data():
    docs, queries, qrels = load_cranfield()
    inverted_index, doc_lengths, avg_dl, N, tf_raw, vocabulary = build_inverted_index(docs)
    kgram_index = build_kgram_index(vocabulary, k=2)
    set_doc_frequencies(inverted_index, vocabulary)
    return docs, queries, qrels, inverted_index, doc_lengths, avg_dl, N, tf_raw, vocabulary, kgram_index


docs, queries_data, qrels, inverted_index, doc_lengths, avg_dl, N, tf_raw, vocabulary, kgram_index = load_data()

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Search bar styling */
    .stTextInput > div > div > input {
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
    }
    
    /* Result card */
    .result-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 4px solid #1d3557;
        border-radius: 6px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .result-card:hover {
        transform: translateX(4px);
    }
    .result-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .result-snippet {
        font-size: 0.9rem;
        color: #495057;
        line-height: 1.5;
    }
    .result-score {
        font-size: 0.8rem;
        color: #6c757d;
        font-family: monospace;
        margin-top: 0.3rem;
    }
    .result-rank {
        display: inline-block;
        background: #1d3557;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        text-align: center;
        line-height: 24px;
        font-size: 0.75rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    /* Meta line */
    .meta-line {
        font-size: 0.85rem;
        color: #6c757d;
        margin-bottom: 1rem;
    }
    
    /* Did you mean */
    .did-you-mean {
        color: #dc3545;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        color: #1d3557;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — Control Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Control Panel")
    st.markdown("---")

    # Model selector
    model_name = st.selectbox(
        "Ranking Model",
        list(MODELS.keys()),
        index=1,  # Default to BM25
        help="Select the ranking algorithm to use."
    )

    # Dynamic sliders based on model
    model_params = {}
    if model_name == "BM25":
        default_k1 = best_params["BM25"]["k1"]
        default_b = best_params["BM25"]["b"]
        model_params["k1"] = st.slider(
            "k₁ (term frequency saturation)",
            min_value=0.0, max_value=3.0,
            value=float(default_k1), step=0.05,
        )
        model_params["b"] = st.slider(
            "b (document length normalization)",
            min_value=0.0, max_value=1.0,
            value=float(default_b), step=0.05,
        )
    elif model_name == "QLM (JM)":
        default_lam = best_params["QLM (JM)"]["lam"]
        model_params["lam"] = st.slider(
            "λ (Jelinek-Mercer smoothing)",
            min_value=0.0, max_value=1.0,
            value=float(default_lam), step=0.05,
        )
    elif model_name == "QLM (Dirichlet)":
        default_mu = best_params["QLM (Dirichlet)"]["mu"]
        model_params["mu"] = st.slider(
            "μ (Dirichlet prior)",
            min_value=0.0, max_value=5000.0,
            value=float(default_mu), step=50.0,
        )

    st.markdown("---")

    # Spell-check toggle
    spell_check_enabled = st.checkbox(
        "Enable Spell Correction",
        value=True,
        help="Automatically correct misspelled query terms using K-gram index and edit distance."
    )

    # Number of results
    top_k_display = st.slider(
        "Results to display",
        min_value=5, max_value=50, value=10, step=5,
    )

    st.markdown("---")
    st.caption(f"Corpus: Cranfield ({N} docs)")
    st.caption(f"Vocabulary: {len(inverted_index)} terms")
    if os.path.exists(BEST_PARAMS_FILE):
        st.caption("Using tuned parameters")
    else:
        st.caption("Using default parameters")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Cranfield IR System</h1>
</div>
""", unsafe_allow_html=True)

# Search box
query_text = st.text_input(
    "Enter your search query",
    placeholder="e.g., boundary layer theory for compressible flow",
    label_visibility="collapsed",
)

if query_text.strip():
    # Spell correction
    corrected_text = query_text
    was_corrected = False
    if spell_check_enabled:
        corrected_text, was_corrected = correct_query(query_text, kgram_index, vocabulary, k=2)

    if was_corrected:
        st.markdown(
            f'<div class="did-you-mean">Did you mean: <b>{corrected_text}</b>?</div>',
            unsafe_allow_html=True,
        )

    # Preprocess and rank
    search_text = corrected_text if spell_check_enabled else query_text
    q_tokens = preprocess(search_text)

    t_start = time.perf_counter()
    results = rank(
        model_name, q_tokens, inverted_index, tf_raw,
        doc_lengths, avg_dl, N, top_k=top_k_display, **model_params
    )
    t_elapsed_ms = (time.perf_counter() - t_start) * 1000

    # Meta line
    st.markdown(
        f'<div class="meta-line">Found <b>{len(results)}</b> results in <b>{t_elapsed_ms:.1f} ms</b> '
        f'using <b>{model_name}</b></div>',
        unsafe_allow_html=True,
    )

    # Display results
    if not results:
        st.info("No results found. Try a different query or adjust parameters.")
    else:
        for i, (doc_id, score) in enumerate(results, start=1):
            doc = docs[doc_id]
            title = doc["title"].strip()
            body = doc["body"].strip()
            # Create snippet: first 200 chars of body
            snippet = body[:250].replace("\n", " ").strip()
            if len(body) > 250:
                snippet += "..."

            st.markdown(
                f"""<div class="result-card">
                    <div class="result-title">
                        <span class="result-rank">{i}</span>
                        {title} <span style="color:#adb5bd; font-size:0.8rem;">(Doc #{doc_id})</span>
                    </div>
                    <div class="result-snippet">{snippet}</div>
                    <div class="result-score">Score: {score:.6f}</div>
                </div>""",
                unsafe_allow_html=True,
            )

else:
    # Show example queries when search is empty
    st.markdown("### Try one of these queries:")
    example_qids = [1, 5, 10, 30, 50]
    cols = st.columns(len(example_qids))
    for col, qid in zip(cols, example_qids):
        with col:
            q = queries_data[qid]
            short = q[:50] + "..." if len(q) > 50 else q
            st.caption(f"**Q{qid}**: {short}")
