
import arxiv
import json
import re
import time
import requests
import pymupdf
import pandas as pd

from pathlib import Path
from datetime import datetime, timezone


# ==========================================
# 1. PATHS AND SETTINGS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PDF_DIR = ROOT / "data" / "pdfs"
TEXT_DIR = ROOT / "data" / "processed" / "texts"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "papers_metadata.json"
MANIFEST_FILE = RAW_DIR / "collection_manifest.json"

TARGET_REPLACEMENTS = 4
MAX_CANDIDATES = 200
DELAY = 3

TOPIC = "Knowledge Representation and Reasoning"

DATE_RANGE = "submittedDate:[202101010000 TO 202610312359]"

TOPIC_QUERY = (
    'all:"knowledge representation" OR '
    'all:"automated reasoning" OR '
    'all:"knowledge reasoning"'
)

HEADERS = {
    "User-Agent": "AcademicResearchCorpus/1.0 (educational project)"
}


# ==========================================
# 2. LOAD EXISTING DATA
# ==========================================

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    papers = json.load(f)

existing_ids = {
    p["arxiv_id"] for p in papers
}

print("Existing metadata records:", len(papers))


# ==========================================
# 3. ARXIV CLIENT
# ==========================================

client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=5
)

query = (
    f"cat:cs.AI AND "
    f"({TOPIC_QUERY}) AND "
    f"{DATE_RANGE}"
)

search = arxiv.Search(
    query=query,
    max_results=MAX_CANDIDATES,
    sort_by=arxiv.SortCriterion.Relevance,
    sort_order=arxiv.SortOrder.Descending
)


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def normalize_id(raw_id):
    return re.sub(r"v\d+$", "", raw_id)


def safe_filename(arxiv_id):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", arxiv_id)


def classify_length(pages):

    if pages < 6:
        return "Short"

    if 7 <= pages <= 9:
        return "Medium"

    if pages > 9:
        return "Long"

    return "Unclassified_6_Pages"


def save_metadata():

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(
            papers,
            f,
            ensure_ascii=False,
            indent=4
        )

    pd.DataFrame(papers).to_csv(
        RAW_DIR / "papers_metadata.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ==========================================
# 5. DOWNLOAD AND VERIFY PDF
# ==========================================

def download_and_extract(pdf_url, pdf_path):

    response = requests.get(
        pdf_url,
        headers=HEADERS,
        timeout=90
    )

    response.raise_for_status()

    content = response.content

    if not content.startswith(b"%PDF-"):
        raise ValueError("The URL did not return a valid PDF.")

    with open(pdf_path, "wb") as f:
        f.write(content)

    document = pymupdf.open(pdf_path)

    page_count = len(document)

    full_text = "\n\n".join(
        page.get_text("text")
        for page in document
    )

    document.close()

    if not full_text.strip():
        raise ValueError("PDF has no extractable text.")

    return full_text, page_count


# ==========================================
# 6. FIND AND ADD REPLACEMENTS
# ==========================================

added = 0
checked = 0
replacement_log = []

print("\nSearching for replacement papers...")
print("Target topic:", TOPIC)

for paper in client.results(search):

    if added >= TARGET_REPLACEMENTS:
        break

    paper_id = normalize_id(paper.get_short_id())

    # Avoid duplicates
    if paper_id in existing_ids:
        continue

    checked += 1

    filename = safe_filename(paper_id)

    pdf_path = PDF_DIR / f"{filename}.pdf"
    text_path = TEXT_DIR / f"{filename}.txt"

    print(f"\nChecking candidate: {paper_id}")
    print("Title:", paper.title)

    try:

        # Download and validate PDF
        full_text, page_count = download_and_extract(
            paper.pdf_url,
            pdf_path
        )

        # Save extracted text
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(full_text)

        # Create metadata record
        new_record = {

            "arxiv_id": paper_id,

            "title": paper.title,

            "abstract": paper.summary,

            "arxiv_url": f"https://arxiv.org/abs/{paper_id}",

            "pdf_url": f"https://arxiv.org/pdf/{paper_id}",

            "published": (
                paper.published.isoformat()
                if paper.published else None
            ),

            "updated": (
                paper.updated.isoformat()
                if paper.updated else None
            ),

            "authors": [
                author.name for author in paper.authors
            ],

            "categories": paper.categories,

            "primary_subtopic": TOPIC,

            "matched_subtopics": [TOPIC],

            "pdf_path": str(pdf_path),

            "text_path": str(text_path),

            "page_count": page_count,

            "length_category": classify_length(page_count),

            "full_text": None,

            "extraction_status": "success",

            "text_characters": len(full_text)
        }

        papers.append(new_record)
        existing_ids.add(paper_id)

        added += 1

        replacement_log.append({
            "arxiv_id": paper_id,
            "title": paper.title,
            "page_count": page_count,
            "status": "success"
        })

        print("SUCCESS: PDF downloaded and text extracted.")
        print("Pages:", page_count)
        print("Replacement progress:", added, "/", TARGET_REPLACEMENTS)

        save_metadata()

    except Exception as error:

        # Remove incomplete files if they exist
        if pdf_path.exists():
            pdf_path.unlink()

        if text_path.exists():
            text_path.unlink()

        print("SKIPPED:", error)

        replacement_log.append({
            "arxiv_id": paper_id,
            "title": paper.title,
            "status": "failed",
            "error": str(error)
        })

    time.sleep(DELAY)


# ==========================================
# 7. UPDATE MANIFEST
# ==========================================

with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
    manifest = json.load(f)

topic_counts = {}

for p in papers:

    topic = p["primary_subtopic"]

    topic_counts[topic] = topic_counts.get(topic, 0) + 1

successful_count = sum(
    1 for p in papers
    if p.get("extraction_status") == "success"
)

manifest["topic_counts"] = topic_counts
manifest["total_unique_papers"] = len(papers)
manifest["successful_extractions"] = successful_count
manifest["metadata_complete"] = len(papers) >= 400
manifest["collection_updated_utc"] = datetime.now(
    timezone.utc
).isoformat()

with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
    json.dump(
        manifest,
        f,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 8. SAVE REPLACEMENT LOG
# ==========================================

with open(
    RAW_DIR / "replacement_log.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        replacement_log,
        f,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 9. FINAL REPORT
# ==========================================

save_metadata()

print("\n" + "=" * 60)
print("REPLACEMENT SUMMARY")
print("=" * 60)

print("Candidates checked:", checked)
print("Successful replacements:", added)
print("Total metadata records:", len(papers))
print("Successful text extractions:", successful_count)

print("\nTopic distribution:")

for topic, count in topic_counts.items():
    print(f"{topic}: {count}")

if added == TARGET_REPLACEMENTS:
    print("\nSUCCESS: All four replacement papers were added.")
else:
    print("\nWARNING: Not all four replacements were found.")

print("\nReplacement log saved.")
print("Metadata and manifest updated.")
