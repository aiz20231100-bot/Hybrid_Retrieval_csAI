from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
INDEXES = DATA / "indexes"
CORPUS_JSON = PROCESSED / "final_corpus.json"
TEXTS_DIR = PROCESSED / "texts"
INDEXES.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150    # حجم كل chunk بالكلمات (يناسب حد النموذج 256 token)
CHUNK_STRIDE = 120   # الخطوة بين chunk والتالي (تداخل 30 كلمة)