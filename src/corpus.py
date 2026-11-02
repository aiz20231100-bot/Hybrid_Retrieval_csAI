import json
import re
from pathlib import Path

from config import CORPUS_JSON, TEXTS_DIR, ROOT


def _pick(rec, names, default=""):
    for n in names:
        v = rec.get(n)
        if v not in (None, "", []):
            return v
    return default


def _read_text_file(rec, arxiv_id):
    path_val = _pick(rec, ["text_path", "text_file", "txt_path"])
    if path_val:
        p = Path(path_val)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists():
            return p.read_text(encoding="utf-8", errors="ignore")
    for p in sorted(TEXTS_DIR.glob(f"{arxiv_id}*")):
        return p.read_text(encoding="utf-8", errors="ignore")
    return ""


def load_corpus():
    with open(CORPUS_JSON, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = _pick(data, ["papers", "records", "data"], list(data.values()))

    docs = []
    for rec in data:
        raw_id = str(_pick(rec, ["arxiv_id", "id", "paper_id", "short_id"]))
        aid = re.sub(r"v\d+$", "", raw_id.split("/abs/")[-1])
        text = _pick(rec, ["full_text", "text", "content"]) or _read_text_file(rec, aid)
        docs.append({
            "arxiv_id": aid,
            "title": str(_pick(rec, ["title"])),
            "abstract": str(_pick(rec, ["abstract", "summary"])),
            "url": str(_pick(rec, ["arxiv_url", "url", "link", "entry_id"],
                             f"https://arxiv.org/abs/{aid}")),
            "subtopic": str(_pick(rec, ["primary_subtopic", "subtopic"])),
            "matched_subtopics": rec.get("matched_subtopics", []),
            "page_count": rec.get("page_count"),            "length_category": str(_pick(rec, ["length_category", "length_type"])),
            "text": str(text),
        })
    return docs


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from collections import Counter

    with open(CORPUS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    first = raw[0] if isinstance(raw, list) else next(iter(raw.values()))
    if isinstance(first, list):
        first = first[0]
    print("Raw keys in first record:", list(first.keys()))

    docs = load_corpus()
    empty = [d["arxiv_id"] for d in docs if len(d["text"].split()) < 200]
    words = [len(d["text"].split()) for d in docs]
    print(f"Documents: {len(docs)}")
    print(f"Empty/very short texts: {len(empty)} {empty[:5]}")
    print(f"Missing title: {sum(not d['title'] for d in docs)} | "
          f"missing abstract: {sum(not d['abstract'] for d in docs)} | "
          f"missing subtopic: {sum(not d['subtopic'] for d in docs)}")
    print(f"Avg words per paper: {sum(words) / len(words):.0f}")
    print("Subtopics:", dict(Counter(d["subtopic"] for d in docs)))
    print("Sample:", docs[0]["arxiv_id"], "|", docs[0]["title"][:80])