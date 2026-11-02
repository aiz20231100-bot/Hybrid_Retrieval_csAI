import time

from corpus import load_corpus
from bm25_retriever import BM25Retriever
from dense_retriever import DenseRetriever


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} papers")

    t = time.time()
    bm25 = BM25Retriever(docs)
    bm25.build()
    bm25.save()
    print(f"BM25 index saved ({time.time() - t:.0f}s)")

    t = time.time()
    dense = DenseRetriever(docs)
    dense.build()
    dense.save()
    print(f"Dense index saved ({time.time() - t:.0f}s)")
    print("Done. The spelling vocabulary is built automatically on first search.")


if __name__ == "__main__":
    main()