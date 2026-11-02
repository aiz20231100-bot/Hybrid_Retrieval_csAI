
import json
import pandas as pd

from pathlib import Path
from collections import Counter


# ==========================================
# 1. PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = RAW_DIR / "papers_metadata.json"

FINAL_JSON = OUTPUT_DIR / "final_corpus.json"
FINAL_CSV = OUTPUT_DIR / "final_corpus.csv"
REPORT_FILE = OUTPUT_DIR / "corpus_validation_report.json"


# ==========================================
# 2. LOAD ORIGINAL METADATA
# ==========================================

with open(SOURCE_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)


# ==========================================
# 3. VALIDATE EACH PAPER
# ==========================================

valid_papers = []
invalid_papers = []

seen_ids = set()

for paper in papers:

    paper_id = paper.get("arxiv_id")

    pdf_path = Path(paper.get("pdf_path") or "")
    text_path = Path(paper.get("text_path") or "")

    valid = (
        paper.get("extraction_status") == "success"
        and paper.get("page_count") is not None
        and paper.get("full_text") is None
        and pdf_path.is_file()
        and text_path.is_file()
        and text_path.stat().st_size > 0
        and paper_id not in seen_ids
    )

    if valid:

        seen_ids.add(paper_id)
        valid_papers.append(paper)

    else:

        invalid_papers.append({
            "arxiv_id": paper_id,
            "title": paper.get("title"),
            "reason": "Failed validation or duplicate"
        })


# ==========================================
# 4. CALCULATE STATISTICS
# ==========================================

topic_counts = Counter(
    p["primary_subtopic"] for p in valid_papers
)

length_counts = Counter(
    p["length_category"] for p in valid_papers
)

total_pages = sum(
    p["page_count"] for p in valid_papers
)

average_pages = (
    total_pages / len(valid_papers)
    if valid_papers else 0
)


# ==========================================
# 5. SAVE CLEAN CORPUS
# ==========================================

with open(FINAL_JSON, "w", encoding="utf-8") as file:

    json.dump(
        valid_papers,
        file,
        ensure_ascii=False,
        indent=4
    )


pd.DataFrame(valid_papers).to_csv(
    FINAL_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================
# 6. VALIDATION REPORT
# ==========================================

report = {

    "total_source_records": len(papers),

    "valid_unique_papers": len(valid_papers),

    "invalid_records": len(invalid_papers),

    "unique_arxiv_ids": len(seen_ids),

    "topic_distribution": dict(topic_counts),

    "length_distribution": dict(length_counts),

    "total_pages": total_pages,

    "average_pages": round(average_pages, 2),

    "target_reached": len(valid_papers) >= 400,

    "all_four_topics_have_100": (
        len(topic_counts) == 4
        and all(count == 100 for count in topic_counts.values())
    ),

    "invalid_papers": invalid_papers
}


with open(REPORT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        report,
        file,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 7. PRINT RESULTS
# ==========================================

print("\n" + "=" * 65)
print("FINAL CORPUS VALIDATION")
print("=" * 65)

print("Original metadata records:", len(papers))

print("Valid unique papers:", len(valid_papers))

print("Invalid or duplicate records:", len(invalid_papers))

print("\nTopic distribution:")

for topic, count in topic_counts.items():
    print(f"{topic}: {count}")

print("\nDocument length distribution:")

for category in [
    "Short",
    "Medium",
    "Long",
    "Unclassified_6_Pages"
]:

    print(f"{category}: {length_counts.get(category, 0)}")

print("\nTotal pages:", total_pages)

print("Average pages:", round(average_pages, 2))

print("\nFinal corpus JSON:", FINAL_JSON)

print("Final corpus CSV:", FINAL_CSV)

print("Validation report:", REPORT_FILE)

if report["target_reached"] and report["all_four_topics_have_100"]:

    print("\nSUCCESS: Final corpus contains 400 valid papers.")

else:

    print("\nWARNING: Corpus needs further inspection.")
