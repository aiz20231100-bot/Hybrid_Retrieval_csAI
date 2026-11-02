import pickle

import numpy as np
from rank_bm25 import BM25Okapi

from config import INDEXES
from preprocessing import clean_text, tokenize

BM25_PATH = INDEXES / "bm25.pkl"


class BM25Retriever:
    def __init__(self, docs):
        self.docs = docs
        self.bm25 = None

    def build(self):
        corpus_tokens = [
            tokenize(f"{d['title']} {d['abstract']} {clean_text(d['text'])}")
            for d in self.docs
        ]
        self.bm25 = BM25Okapi(corpus_tokens)  # k1=1.5, b=0.75 (القيم الافتراضية)

    def save(self):
        with open(BM25_PATH, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self):
        with open(BM25_PATH, "rb") as f:
            self.bm25 = pickle.load(f)

    def search(self, query, top_k=10):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]