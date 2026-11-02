import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from search_engine import SearchEngine  # noqa: E402

st.set_page_config(page_title="cs.AI Hybrid Retrieval", page_icon="🔎", layout="wide")

MODES = {
    "BM25 (sparse)": "bm25",
    "Dense (embeddings)": "dense",
    "Hybrid (BM25 + Dense, RRF fusion)": "hybrid",
}


@st.cache_resource(show_spinner="Loading indexes and models (first time only)...")
def get_engine():
    return SearchEngine()


def esc(text):
    """يمنع Streamlit من تفسير الرموز [ ] $ كـ Markdown/LaTeX."""
    return (text.replace("[", "\\[").replace("]", "\\]").replace("$", "\\$"))


engine = get_engine()

with st.sidebar:
    st.header("Settings")
    mode_label = st.radio("Retrieval / fusion method", list(MODES))
    mode = MODES[mode_label]
    rrf_k = st.slider("RRF constant k", 1, 100, 60, disabled=(mode != "hybrid"),
                      help="score = sum of 1 / (k + rank) over BM25 and Dense rankings")
    st.markdown("---")
    st.caption(f"Corpus: {len(engine.docs)} cs.AI papers, 4 sub-topics.")
    st.caption('Tip: put a phrase in quotes for exact matching, e.g. "multi-agent systems".')

st.title("🔎 Hybrid Retrieval over arXiv cs.AI")
st.caption("BM25 + Dense Retrieval + Reciprocal Rank Fusion")

with st.form("search_form"):
    query = st.text_input("Search query", placeholder="e.g. heuristic search for planning")
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    start = time.perf_counter()
    results, info = engine.search(query, mode=mode, top_k=10, rrf_k=rrf_k)
    elapsed = time.perf_counter() - start

    if info["corrections"]:
        st.info("Spelling corrected: " +
                ", ".join(f"{a} → {b}" for a, b in info["corrections"]))
    if info["expansions"]:
        st.caption("Query expansion: " +
                   ", ".join(f"{a} → {b}" for a, b in info["expansions"]))
    if info["phrases"]:
        st.caption(f"Exact phrase filter: {info['phrases']} "
                   f"({info.get('phrase_matches', 0)} papers contain it)")

    if not results:
        st.warning("No results found.")
    else:
        st.success(f"Top {len(results)} results · {mode_label} · {elapsed:.2f}s")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r['rank']}. [{esc(r['title'])}]({r['url']})**")
                meta = (f"{r['subtopic']} · arXiv:{r['arxiv_id']} · "
                        f"{r['score_name']} score: {r['score']:.4f}")
                if mode == "hybrid":
                    meta += (f" · BM25 rank: {r['bm25_rank'] or '–'}"
                             f" · Dense rank: {r['dense_rank'] or '–'}")
                st.caption(meta)
                st.write(esc(r["snippet"]))
                st.markdown(f"[{r['url']}]({r['url']})")
elif submitted:
    st.warning("Please enter a query.")