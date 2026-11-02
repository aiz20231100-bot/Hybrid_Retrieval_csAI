import sys
sys.stdout.reconfigure(encoding="utf-8")

from corpus import load_corpus
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever
from fusion import rrf_fuse

QUERIES = [
    "heuristic search for automated planning",
    "cooperative multi-agent reinforcement learning",
    "description logic ontology reasoning",
    "embodied AI robot manipulation",
    "planning with large language models",
]


def main():
    docs = load_corpus()
    bm25 = BM25Retriever(docs)
    bm25.load()
    dense = DenseRetriever(docs)
    dense.load()

    for q in QUERIES:
        b = bm25.search(q, top_k=100)
        d = dense.search(q, top_k=100)
        b_rank = {i: r for r, (i, _) in enumerate(b, 1)}
        d_rank = {i: r for r, (i, _) in enumerate(d, 1)}
        fused = rrf_fuse([b, d], k=60, top_k=5)

        print(f"\n--- HYBRID (RRF) | {q}")
        for rank, (i, score) in enumerate(fused, 1):
            print(f"{rank:>2}. [RRF {score:.4f}] {docs[i]['title'][:80]}")
            print(f"      BM25 rank: {b_rank.get(i, '-')} | "
                  f"Dense rank: {d_rank.get(i, '-')} | {docs[i]['subtopic']}")


if __name__ == "__main__":
    main()