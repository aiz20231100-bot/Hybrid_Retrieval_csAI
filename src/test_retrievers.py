import sys
sys.stdout.reconfigure(encoding="utf-8")

from corpus import load_corpus
from bm25_retriever import BM25Retriever, BM25_PATH
from dense_retriever import DenseRetriever, FAISS_PATH

QUERIES = [
    "heuristic search for automated planning",
    "cooperative multi-agent reinforcement learning",
    "description logic ontology reasoning",
    "embodied AI robot manipulation",
]


def show(title, results, docs):
    print(f"\n--- {title}")
    for rank, (i, score) in enumerate(results, 1):
        d = docs[i]
        print(f"{rank:>2}. [{score:.4f}] {d['title'][:90]}")
        print(f"      {d['subtopic']} | {d['arxiv_id']}")


def run(which):
    docs = load_corpus()
    if which == "bm25":
        r = BM25Retriever(docs)
        if BM25_PATH.exists():
            r.load()
        else:
            print("Building BM25 index...")
            r.build()
            r.save()
    else:
        r = DenseRetriever(docs)
        if FAISS_PATH.exists():
            r.load()
        else:
            print("Building dense index (several minutes on CPU)...")
            r.build()
            r.save()

    for q in QUERIES:
        show(f"{which.upper()} | {q}", r.search(q, top_k=5), docs)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "bm25")