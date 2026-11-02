
import json
import time
import re
import requests
import fitz

from pathlib import Path
from tqdm import tqdm


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "texts"

PDF_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = RAW_DIR / "papers_metadata.json"
OUTPUT_FILE = RAW_DIR / "papers_metadata.json"
ERROR_FILE = RAW_DIR / "download_errors.json"


# ==========================================
# 2. DOWNLOAD SETTINGS
# ==========================================

DELAY_SECONDS = 3
MAX_RETRIES = 4
SAVE_EVERY = 5

HEADERS = {
    "User-Agent": "AcademicResearchCorpus/1.0 (educational project)"
}


# ==========================================
# 3. LOAD METADATA
# ==========================================

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    papers = json.load(file)

print(f"Total papers: {len(papers)}")


# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================

def safe_filename(arxiv_id):
    return re.sub(r"[^a-zA-Z0-9._-]", "_", arxiv_id)


def classify_length(page_count):

    if page_count < 6:
        return "Short"

    elif 7 <= page_count <= 9:
        return "Medium"

    elif page_count > 9:
        return "Long"

    else:
        # Exactly 6 pages is not defined in the assignment.
        return "Unclassified_6_Pages"


def save_metadata():

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(
            papers,
            file,
            ensure_ascii=False,
            indent=4
        )


def save_errors(errors):

    with open(ERROR_FILE, "w", encoding="utf-8") as file:
        json.dump(
            errors,
            file,
            ensure_ascii=False,
            indent=4
        )


# ==========================================
# 5. DOWNLOAD PDF
# ==========================================

def download_pdf(paper, pdf_path):

    pdf_url = paper["pdf_url"] + "?download=1"

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                pdf_url,
                headers=HEADERS,
                timeout=90
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            content = response.content

            # Verify that the response is actually a PDF.
            if not content.startswith(b"%PDF-"):
                raise ValueError(
                    f"Response is not a valid PDF. "
                    f"Content-Type: {content_type}"
                )

            with open(pdf_path, "wb") as file:
                file.write(content)

            return True

        except Exception as error:

            print(
                f"\nDownload attempt {attempt} failed "
                f"for {paper['arxiv_id']}: {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(5 * attempt)

    return False


# ==========================================
# 6. EXTRACT TEXT AND PAGE COUNT
# ==========================================

def extract_pdf_text(pdf_path):

    document = fitz.open(pdf_path)

    page_count = len(document)

    pages_text = []

    for page in document:

        text = page.get_text("text")

        pages_text.append(text)

    full_text = "\n\n".join(pages_text)

    document.close()

    return full_text, page_count


# ==========================================
# 7. PROCESS PAPERS
# ==========================================

errors = []

stats = {
    "downloaded": 0,
    "extracted": 0,
    "failed": 0,
    "empty_text": 0
}

for index, paper in enumerate(
    tqdm(papers, desc="Processing papers"),
    start=1
):

    paper_id = paper["arxiv_id"]

    filename = safe_filename(paper_id)

    pdf_path = PDF_DIR / f"{filename}.pdf"
    text_path = TEXT_DIR / f"{filename}.txt"

    # Skip papers already processed successfully.
    if (
        paper.get("extraction_status") == "success"
        and text_path.exists()
        and pdf_path.exists()
    ):

        stats["downloaded"] += 1
        stats["extracted"] += 1
        continue

    try:

        # ------------------------------
        # Download
        # ------------------------------

        if not pdf_path.exists():

            success = download_pdf(
                paper,
                pdf_path
            )

            if not success:

                raise RuntimeError(
                    "PDF download failed after all retries"
                )

        stats["downloaded"] += 1

        # ------------------------------
        # Extract
        # ------------------------------

        full_text, page_count = extract_pdf_text(
            pdf_path
        )

        if not full_text.strip():

            paper["extraction_status"] = "empty_text"

            stats["empty_text"] += 1

            raise ValueError(
                "PDF contains no extractable text"
            )

        # ------------------------------
        # Save extracted text
        # ------------------------------

        with open(
            text_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(full_text)

        # ------------------------------
        # Update metadata
        # ------------------------------

        paper["pdf_path"] = str(pdf_path)

        paper["text_path"] = str(text_path)

        paper["page_count"] = page_count

        paper["length_category"] = classify_length(
            page_count
        )

        paper["full_text"] = None

        paper["extraction_status"] = "success"

        paper["text_characters"] = len(full_text)

        stats["extracted"] += 1

    except Exception as error:

        paper["extraction_status"] = "failed"

        stats["failed"] += 1

        errors.append({
            "arxiv_id": paper_id,
            "title": paper["title"],
            "error": str(error)
        })

        print(f"\nERROR: {paper_id}: {error}")

    finally:

        # Save progress periodically.
        if index % SAVE_EVERY == 0:

            save_metadata()
            save_errors(errors)

        time.sleep(DELAY_SECONDS)


# ==========================================
# 8. FINAL SAVE
# ==========================================

save_metadata()
save_errors(errors)


# ==========================================
# 9. FINAL STATISTICS
# ==========================================

length_counts = {
    "Short": 0,
    "Medium": 0,
    "Long": 0,
    "Unclassified_6_Pages": 0
}

successful_papers = [
    p for p in papers
    if p.get("extraction_status") == "success"
]

for paper in successful_papers:

    category = paper.get("length_category")

    if category in length_counts:
        length_counts[category] += 1


print("\n" + "=" * 65)
print("PDF PROCESSING SUMMARY")
print("=" * 65)

print(f"Total papers: {len(papers)}")

print(f"Downloaded: {stats['downloaded']}")

print(f"Successfully extracted: {stats['extracted']}")

print(f"Failed: {stats['failed']}")

print(f"Empty text: {stats['empty_text']}")

print("\nDocument length distribution:")

for category, count in length_counts.items():

    print(f"{category}: {count}")

print("\nFiles saved:")

print(f"PDF directory: {PDF_DIR}")

print(f"Text directory: {TEXT_DIR}")

print(f"Metadata: {OUTPUT_FILE}")

print(f"Errors: {ERROR_FILE}")

print("\nProcessing completed.")
