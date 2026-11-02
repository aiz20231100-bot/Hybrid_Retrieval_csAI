
import arxiv
import pandas as pd
import json
import re
import time

from pathlib import Path
from datetime import datetime, timezone


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PER_TOPIC = 100
MAX_RESULTS_PER_TOPIC = 300

DATE_RANGE = (
    "submittedDate:[202101010000 TO 202610312359]"
)


# ==========================================
# 2. SUB-TOPIC SEARCH QUERIES
# ==========================================

TOPICS = {
    "Search and Planning": (
        'all:"automated planning" OR '
        'all:"heuristic search" OR '
        'all:"search-based planning"'
    ),

    "Knowledge Representation and Reasoning": (
        'all:"knowledge representation" OR '
        'all:"automated reasoning" OR '
        'all:"knowledge reasoning"'
    ),

    "Multi-Agent Systems": (
        'all:"multi-agent systems" OR '
        'all:"multi agent" OR '
        'all:"multiagent systems"'
    ),

    "Robotics and Embodied AI": (
        'all:"embodied AI" OR '
        'all:"robot learning" OR '
        'all:"robotic planning"'
    )
}


# ==========================================
# 3. ARXIV CLIENT
# ==========================================

client = arxiv.Client(
    page_size=100,
    delay_seconds=3,
    num_retries=5
)


# ==========================================
# 4. NORMALIZE ARXIV ID
# ==========================================

def normalize_arxiv_id(raw_id):

    return re.sub(r"v\d+$", "", raw_id)


# ==========================================
# 5. COLLECT PAPERS
# ==========================================

papers = {}

topic_counts = {}

queries_used = {}


for topic_name, topic_terms in TOPICS.items():

    print("\n" + "=" * 70)
    print(f"COLLECTING: {topic_name}")
    print("=" * 70)

    query = (
        f"cat:cs.AI AND "
        f"({topic_terms}) AND "
        f"{DATE_RANGE}"
    )

    queries_used[topic_name] = query

    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS_PER_TOPIC,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )

    assigned_count = 0

    try:

        for paper in client.results(search):

            raw_id = paper.get_short_id()
            paper_id = normalize_arxiv_id(raw_id)

            # Check if paper already exists
            if paper_id in papers:

                if topic_name not in papers[paper_id][
                    "matched_subtopics"
                ]:

                    papers[paper_id][
                        "matched_subtopics"
                    ].append(topic_name)

                continue

            # Save new unique paper
            papers[paper_id] = {

                "arxiv_id": paper_id,

                "title": paper.title,

                "abstract": paper.summary,

                "arxiv_url": (
                    f"https://arxiv.org/abs/{paper_id}"
                ),

                "pdf_url": (
                    f"https://arxiv.org/pdf/{paper_id}"
                ),

                "published": (
                    paper.published.isoformat()
                    if paper.published else None
                ),

                "updated": (
                    paper.updated.isoformat()
                    if paper.updated else None
                ),

                "authors": [
                    author.name
                    for author in paper.authors
                ],

                "categories": paper.categories,

                "primary_subtopic": topic_name,

                "matched_subtopics": [topic_name],

                "pdf_path": None,

                "page_count": None,

                "length_category": None,

                "full_text": None
            }

            assigned_count += 1

            if assigned_count % 10 == 0:

                print(
                    f"{topic_name}: "
                    f"{assigned_count}/{TARGET_PER_TOPIC}"
                )

            if assigned_count >= TARGET_PER_TOPIC:
                break

    except Exception as error:

        print(f"ERROR collecting {topic_name}: {error}")

    topic_counts[topic_name] = assigned_count

    print(
        f"Finished {topic_name}: "
        f"{assigned_count} unique papers assigned"
    )


# ==========================================
# 6. SAVE DATA
# ==========================================

paper_list = list(papers.values())

df = pd.DataFrame(paper_list)

csv_path = OUTPUT_DIR / "papers_metadata.csv"

json_path = OUTPUT_DIR / "papers_metadata.json"

manifest_path = OUTPUT_DIR / "collection_manifest.json"


df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


with open(json_path, "w", encoding="utf-8") as file:

    json.dump(
        paper_list,
        file,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 7. SAVE REPRODUCIBILITY MANIFEST
# ==========================================

manifest = {

    "category": "cs.AI",

    "date_range": {
        "from": "2021-01-01",
        "to": "2026-10-31"
    },

    "collection_timestamp_utc": (
        datetime.now(timezone.utc).isoformat()
    ),

    "target_per_topic": TARGET_PER_TOPIC,

    "total_unique_papers": len(papers),

    "topic_counts": topic_counts,

    "queries": queries_used,

    "source": "arXiv API",

    "metadata_complete": (
        len(papers) >= 400
    )
}


with open(
    manifest_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        manifest,
        file,
        ensure_ascii=False,
        indent=4
    )


# ==========================================
# 8. FINAL REPORT
# ==========================================

print("\n" + "=" * 70)
print("COLLECTION SUMMARY")
print("=" * 70)

for topic, count in topic_counts.items():

    print(f"{topic}: {count}")

print("-" * 70)

print(f"Total unique papers: {len(papers)}")

print(f"Target: 400")

print(f"CSV saved: {csv_path}")

print(f"JSON saved: {json_path}")

print(f"Manifest saved: {manifest_path}")

if len(papers) >= 400:

    print("\nSUCCESS: At least 400 unique papers collected.")

else:

    print(
        "\nWARNING: Fewer than 400 unique papers collected."
    )

print("\nMetadata collection completed.")
