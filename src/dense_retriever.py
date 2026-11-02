import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import INDEXES, EMBEDDING_MODEL, CHUNK_WORDS, CHUNK_STRIDE
from preprocessing import clean_text

FAISS_PATH = INDEXES / "dense.faiss"
CHUNK_MAP_PATH = INDEXES / "dense_chunk_doc.npy"


def make_chunks(text, size, stride):
    words = text.split()
    chunks = []
    for start in range(0, len(words), stride):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


class DenseRetriever:
    def __init__(self, docs):
        self.docs = docs
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunk_doc = None

    def build(self):
        chunk_texts, chunk_doc = [], []
        for i, d in enumerate(self.docs):
            pieces = [f"{d['title']}. {d['abstract']}"]
            pieces += make_chunks(clean_text(d["text"]), CHUNK_WORDS, CHUNK_STRIDE)
            chunk_texts += pieces
            chunk_doc += [i] * len(pieces)
        print(f"Total chunks to embed: {len(chunk_texts)}")

        emb = self.model.encode(
            chunk_texts, batch_size=64, normalize_embeddings=True,
            show_progress_bar=True, convert_to_numpy=True,
        )
        emb = np.ascontiguousarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.chunk_doc = np.array(chunk_doc, dtype="int32")

    def save(self):
        data = faiss.serialize_index(self.index)
        with open(FAISS_PATH, "wb") as f:
            f.write(data.tobytes())
        np.save(CHUNK_MAP_PATH, self.chunk_doc)

    def load(self):
        with open(FAISS_PATH, "rb") as f:
            buf = np.frombuffer(f.read(), dtype="uint8")
        self.index = faiss.deserialize_index(buf)
        self.chunk_doc = np.load(CHUNK_MAP_PATH)

    def search(self, query, top_k=10, chunk_k=None):
        chunk_k = chunk_k or max(300, top_k * 10)
        q = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        q = np.ascontiguousarray(q, dtype="float32")
        scores, ids = self.index.search(q, chunk_k)
        best = {}
        for s, c in zip(scores[0], ids[0]):
            if c < 0:
                continue
            doc = int(self.chunk_doc[c])
            if doc not in best or s > best[doc]:
                best[doc] = float(s)
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]