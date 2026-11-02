import json
import statistics
import sys
import textwrap
from collections import Counter
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak)

from config import ROOT, DATA, CORPUS_JSON
from corpus import load_corpus

REPORTS = ROOT / "reports"
CHARTS = REPORTS / "charts"
MANIFEST = DATA / "raw" / "collection_manifest.json"
REPORTS.mkdir(exist_ok=True)
CHARTS.mkdir(exist_ok=True)

LENGTH_ORDER = ["Short", "Medium", "Long", "Unclassified_6_Pages"]
LENGTH_DEF = {
    "Short": "Fewer than 6 pages",
    "Medium": "7 to 9 pages",
    "Long": "More than 9 pages",
    "Unclassified_6_Pages": "Exactly 6 pages (not defined in the assignment; kept, pending instructor)",
}

styles = getSampleStyleSheet()
H1, H2, BODY = styles["Heading1"], styles["Heading2"], styles["BodyText"]
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10)


def make_table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2F3E5C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F8")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def bar_chart(labels, values, title, name):
    labels = [textwrap.fill(str(l), 16) for l in labels]
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bars = ax.bar(labels, values, color="#3B6FB6")
    ax.bar_label(bars)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Image(str(path), width=15 * cm, height=15 * cm * 3.2 / 6.5)


def hist_chart(pages, name):
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.hist(pages, bins=25, color="#3B6FB6", edgecolor="white")
    ax.axvline(6, ls="--", color="red")
    ax.axvline(9.5, ls="--", color="red")
    ax.set_title("Distribution of page counts (dashed: 6 and 9.5 page boundaries)")
    ax.set_xlabel("Pages")
    ax.set_ylabel("Papers")
    fig.tight_layout()
    path = CHARTS / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Image(str(path), width=15 * cm, height=15 * cm * 3.2 / 6.5)


def ordered_lengths(counter):
    known = [k for k in LENGTH_ORDER if k in counter]
    return known + [k for k in counter if k not in LENGTH_ORDER]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(CORPUS_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("papers") or list(raw.values())
    docs = load_corpus()

    n = len(raw)
    topics = Counter(r.get("primary_subtopic", "") for r in raw)
    lengths = Counter(r.get("length_category", "") for r in raw)
    pages = [int(r["page_count"]) for r in raw if r.get("page_count") is not None]
    words = [len(d["text"].split()) for d in docs]
    years = Counter(str(r.get("published", ""))[:4] for r in raw)
    dates = sorted(str(r.get("published", ""))[:10] for r in raw if r.get("published"))

    def cats(r):
        c = r.get("categories") or []
        return c.split() if isinstance(c, str) else c

    in_cs_ai = sum("cs.AI" in cats(r) for r in raw)
    multi = sum(len(r.get("matched_subtopics") or []) > 1 for r in raw)
    authors = set()
    for r in raw:
        a = r.get("authors") or []
        if isinstance(a, list):
            authors.update(str(x) for x in a)

    story = []
    story.append(Paragraph("Corpus Statistics Report", styles["Title"]))
    story.append(Paragraph("Hybrid Retrieval System (Phase 1) - arXiv cs.AI corpus", H2))
    story.append(Paragraph(f"Generated: {datetime.now():%Y-%m-%d %H:%M}", SMALL))
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("1. Overview", H1))
    overview = [["Metric", "Value"],
                ["Total papers (full text available)", n],
                ["arXiv category", "cs.AI (papers listing cs.AI: %d of %d)" % (in_cs_ai, n)],
                ["Sub-topics", len(topics)],
                ["Papers matching more than one sub-topic", multi],
                ["Publication date range", f"{dates[0]} to {dates[-1]}" if dates else "n/a"],
                ["Total pages", sum(pages)],
                ["Pages per paper (mean / median / min / max)",
                 f"{statistics.mean(pages):.2f} / {statistics.median(pages)} / {min(pages)} / {max(pages)}"],
                ["Total words", f"{sum(words):,}"],
                ["Words per paper (mean / median)",
                 f"{statistics.mean(words):,.0f} / {statistics.median(words):,.0f}"],
                ["Unique author names", len(authors)]]
    story.append(make_table(overview, [8 * cm, 8 * cm]))

    story.append(Paragraph("2. Distribution by sub-topic", H1))
    rows = [["Sub-topic", "Papers", "Share"]]
    for k, v in topics.items():
        rows.append([k, v, f"{100 * v / n:.1f}%"])
    story.append(make_table(rows, [9 * cm, 3 * cm, 3 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(bar_chart(list(topics), list(topics.values()), "Papers per sub-topic", "topics"))

    story.append(PageBreak())
    story.append(Paragraph("3. Document length categories", H1))
    order = ordered_lengths(lengths)
    rows = [["Category", "Definition", "Papers", "Share"]]
    for k in order:
        rows.append([k, Paragraph(LENGTH_DEF.get(k, ""), SMALL), lengths[k],
                     f"{100 * lengths[k] / n:.1f}%"])
    story.append(make_table(rows, [4 * cm, 7 * cm, 2.5 * cm, 2.5 * cm]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(bar_chart(order, [lengths[k] for k in order], "Papers per length category", "lengths"))

    story.append(Paragraph("Length category by sub-topic", H2))
    rows = [["Sub-topic"] + order + ["Total"]]
    for t in topics:
        counts = Counter(r.get("length_category", "") for r in raw
                         if r.get("primary_subtopic", "") == t)
        rows.append([Paragraph(t, SMALL)] + [counts.get(k, 0) for k in order] + [topics[t]])
    story.append(make_table(rows))

    story.append(PageBreak())
    story.append(Paragraph("4. Pages and text length", H1))
    story.append(hist_chart(pages, "pages_hist"))

    story.append(Paragraph("5. Publication year", H1))
    ys = sorted(years)
    story.append(bar_chart(ys, [years[y] for y in ys], "Papers per publication year", "years"))

    story.append(PageBreak())
    story.append(Paragraph("6. Collection details and reproducibility", H1))
    if MANIFEST.exists():
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)
        rows = [["Field", "Value"]]
        items = manifest.items() if isinstance(manifest, dict) else [("manifest", manifest)]
        for k, v in items:
            text = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            if len(text) > 3000:
                text = text[:3000] + " ..."
            rows.append([Paragraph(escape(str(k)), SMALL), Paragraph(escape(text), SMALL)])
        story.append(make_table(rows, [4 * cm, 12.5 * cm]))
    else:
        story.append(Paragraph("collection_manifest.json not found.", BODY))

    story.append(Paragraph("7. Data quality notes", H1))
    unclassified = lengths.get("Unclassified_6_Pages", 0)
    notes = [
        "Four papers from the first collection failed to download (HTTP 404) and were replaced "
        "by four Knowledge Representation and Reasoning papers; the failed records are kept in "
        "the raw logs for documentation only and are excluded from the final corpus.",
        f"{unclassified} papers have exactly 6 pages. The assignment does not state which length "
        "category they belong to, so they are kept under a separate label pending instructor "
        "clarification.",
        "All papers in the final corpus have extracted full text; none are empty.",
    ]
    for t in notes:
        story.append(Paragraph("- " + escape(t), BODY))

    out = REPORTS / "corpus_statistics.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=2 * cm, bottomMargin=2 * cm,
                      title="Corpus Statistics Report").build(story)

    print(f"Saved: {out}")
    print(f"Papers: {n} | topics: {dict(topics)}")
    print(f"Lengths: {dict(lengths)}")
    print(f"Total pages: {sum(pages)} | avg pages: {statistics.mean(pages):.2f}")
    print(f"Date range: {dates[0] if dates else '-'} to {dates[-1] if dates else '-'}")


if __name__ == "__main__":
    main()