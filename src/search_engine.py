import re

from corpus import load_corpus
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from fusion import rrf_fuse
from preprocessing import clean_text, tokenize
from query_processing import QueryProcessor, normalize


def make_snippet(abstract, query_text, length=200):
    text = re.sub(r"\s+", " ", abstract).strip()
    if len(text) <= length:
        return text
    q = set(tokenize(query_text))
    best_start, best_score, pos = 0, -1, 0
    for s in re.split(r"(?<=[.!?])\s+", text):
        start = text.find(s, pos)
        pos = start + len(s)
        score = len(q & set(tokenize(s)))
        if score > best_score:
            best_score, best_start = score, start
    snippet = text[best_start:best_start + length]
    if best_start + length < len(text):
        snippet = snippet.rsplit(" ", 1)[0] + "..."
    if best_start > 0:
        snippet = "..." + snippet
    return snippet


class SearchEngine:
    MODES = ("bm25", "dense", "hybrid")

    def __init__(self):
        self.docs = load_corpus()
        self.bm25 = BM25Retriever(self.docs)
        self.bm25.load()
        self.dense = DenseRetriever(self.docs)
        self.dense.load()
        self.qp = QueryProcessor(self.docs)
        # نصوص مُطبَّعة لمطابقة العبارات (Phrase Matching)
        self.norm_docs = [
            " " + normalize(f"{d['title']} {d['abstract']} {clean_text(d['text'])}") + " "
            for d in self.docs
        ]

    def search(self, query, mode="hybrid", top_k=10, rrf_k=60):
        info = self.qp.process(query)
        text = info["search_text"]
        if not text:
            return [], info

        phrases = info["phrases"]
        allowed = None
        depth, chunk_k = 100, None
        if phrases:
            allowed = {i for i, t in enumerate(self.norm_docs)
                       if all(f" {p} " in t for p in phrases)}
            info["phrase_matches"] = len(allowed)
            depth, chunk_k = len(self.docs), self.dense.index.ntotal

        rankings = []
        if mode in ("bm25", "hybrid"):
            rankings.append(self.bm25.search(text, top_k=depth))
        if mode in ("dense", "hybrid"):
            rankings.append(self.dense.search(text, top_k=depth, chunk_k=chunk_k))
        if allowed is not None:
            rankings = [[(i, s) for i, s in r if i in allowed] for r in rankings]

        ranks = {}
        if mode == "hybrid":
            ranks = {
                "bm25": {i: r for r, (i, _) in enumerate(rankings[0], 1)},
                "dense": {i: r for r, (i, _) in enumerate(rankings[1], 1)},
            }
            fused, score_name = rrf_fuse(rankings, k=rrf_k, top_k=top_k), "RRF"
        else:
            fused = rankings[0][:top_k]
            score_name = "BM25" if mode == "bm25" else "Cosine"

        results = []
        for rank, (i, score) in enumerate(fused, 1):
            d = self.docs[i]
            results.append({
                "rank": rank, "title": d["title"], "url": d["url"],
                "snippet": make_snippet(d["abstract"], text),
                "score": score, "score_name": score_name,
                "subtopic": d["subtopic"], "arxiv_id": d["arxiv_id"],
                "bm25_rank": ranks.get("bm25", {}).get(i),
                "dense_rank": ranks.get("dense", {}).get(i),
            })
        return results, info