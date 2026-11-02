# Hybrid Retrieval System over arXiv cs.AI (Phase 1)

A hybrid search engine over 400 research papers from arXiv category **cs.AI**. It combines BM25 sparse retrieval and embedding-based dense retrieval, fuses both rankings with Reciprocal Rank Fusion (RRF), and provides a Streamlit search interface.

## Corpus

- Category: cs.AI (Artificial Intelligence)
- 400 papers with full text, 100 per sub-topic: Search and Planning, Knowledge Representation and Reasoning, Multi-Agent Systems, Robotics and Embodied AI
- arXiv query window: 2021-01-01 to 2026-10-31 (papers actually collected: 2021-02-05 to 2026-10-15)
- Length categories (by page count): Short (< 6 pages) 28, Medium (7-9 pages) 52, Long (> 9 pages) 308. Twelve papers have exactly 6 pages; the assignment does not assign them to a category, so they are kept under the separate label `Unclassified_6_Pages`.
- Exact queries, date range and collection time are stored in `data/raw/collection_manifest.json`. A statistics report is in `reports/corpus_statistics.pdf`.
- PDFs are not included in the repository because of their size (about 1.3 GB). Metadata is in `data/raw/` and `data/processed/final_corpus.json`; extracted full texts are in `data/processed/texts/`.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Build the search indexes once (about 15 minutes on CPU, mostly the dense embeddings; the embedding model is downloaded on first use):

```bash
python src/build_indexes.py
```

## Run the search interface

```bash
streamlit run gui/app.py --server.fileWatcherType none
```

The interface provides a search box, a selector for BM25 / Dense / Hybrid (RRF), the top-10 results with title, arXiv link, a snippet of about 200 characters and the ranking score. In Hybrid mode each result also shows its BM25 and Dense ranks.

## How it works

**BM25 (sparse).** Implemented with `rank-bm25` (Okapi BM25, k1 = 1.5, b = 0.75). Each document is title + abstract + full text. Text is lowercased, tokenized, stripped of stop words and stemmed with the Porter stemmer.

**Dense retrieval.** Model `sentence-transformers/all-MiniLM-L6-v2`. Because the model reads at most 256 tokens, each paper is split into overlapping chunks of 150 words (stride 120) plus one title + abstract chunk, giving 35,490 chunks. Embeddings are normalized and stored in a FAISS `IndexFlatIP` index, so inner product equals cosine similarity. A paper's score is the best score among its chunks.

**Fusion (RRF).** Both rankings (top 100 each) are combined with `score(d) = sum over rankers of 1 / (k + rank_r(d))`, k = 60. RRF uses ranks instead of raw scores, which matters because BM25 scores are unbounded while cosine scores lie in [0, 1].

**Query processing.**
- Case insensitivity: all queries and documents are lowercased.
- Spelling correction: edit-distance (up to 2) correction against the vocabulary of the corpus itself, only for words not already in the vocabulary.
- Abbreviation expansion: a domain dictionary (LLM, MARL, PDDL, POMDP, ...) expands in both directions (abbreviation to full form and full form to abbreviation).
- Phrase matching: a phrase in double quotes, for example `"multi-agent reinforcement learning"`, restricts results to papers containing that exact phrase (hyphens and spaces are treated alike).

## Repository structure

```text
data/raw/          metadata, collection manifest, logs
data/processed/    final_corpus.json/.csv, extracted texts, validation report
src/               collection, retrieval, fusion, query processing, search engine
gui/app.py         Streamlit interface
reports/           corpus statistics PDF
ai-logs/phase1/    AI usage disclosure
```

Main modules: `collect_metadata.py`, `download_and_extract.py`, `replace_failed_papers.py`, `validate_final_corpus.py` (data collection); `bm25_retriever.py`, `dense_retriever.py`, `fusion.py`, `query_processing.py`, `search_engine.py` (retrieval); `generate_stats_report.py` (report).

## Known limitations

- The phrase filter searches the full text, including reference lists, so a paper that only cites a phrase in its references can match.
- The 12 six-page papers await instructor clarification on their length category.
- Retrieval quality has been checked qualitatively on sample queries; a quantitative evaluation (Precision@10, MRR, NDCG@10) has not been run.