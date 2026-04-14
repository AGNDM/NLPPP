# scrape_papers.py
import json
import time
from pathlib import Path
from semanticscholar import SemanticScholar

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_FILE = Path("data/papers.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_PER_YEAR = 112   # 9 years × 112 ≈ 1008 total

# Older papers have had more time to accumulate citations, so threshold scales down
MIN_CITATIONS_BY_YEAR = {
    2015: 150, 2016: 120, 2017: 100,
    2018: 80,  2019: 60,  2020: 40,
    2021: 25,  2022: 15,  2023: 5,
}

FIELDS = [
    "paperId", "title", "abstract", "year",
    "authors", "venue", "externalIds",
    "openAccessPdf", "citationCount",
]

# ── Client ────────────────────────────────────────────────────────────────────
# retry=True is the default — the library handles 429s automatically,
# waiting 30s and retrying up to 10 times. No manual sleep needed.
sch = SemanticScholar(retry=True)

# ── Scrape ────────────────────────────────────────────────────────────────────
all_papers: list[dict] = []
seen_ids: set[str] = set()

for year in range(2015, 2024):
    min_cit = MIN_CITATIONS_BY_YEAR[year]
    print(f"\n{'─'*55}")
    print(f"  Year {year}  |  min citations: {min_cit}")
    print(f"{'─'*55}")

    # bulk=True  → uses the bulk search endpoint (much more permissive rate limits)
    # fields_of_study → server-side filter, saves wasted pages
    # venue        → server-side filter for high-quality NLP conferences
    results = sch.search_paper(
        query="natural language processing",
        year=str(year),
        min_citation_count=min_cit,
        fields_of_study=["Computer Science", "Linguistics"],
        venue=[
            "ACL", "EMNLP", "NAACL", "EACL", "CoNLL",
            "Transactions of the Association for Computational Linguistics",
        ],
        fields=FIELDS,
        bulk=True,
        sort="citationCount:desc",  # highest-cited first — best quality at the front
    )

    collected = 0
    for paper in results:  # the library auto-paginates as you iterate
        if collected >= TARGET_PER_YEAR:
            break
        if not paper.abstract:          # skip papers with no abstract
            continue
        if paper.paperId in seen_ids:   # skip global duplicates
            continue

        # Convert the typed Paper object to a plain dict for JSON serialisation
        all_papers.append({
            "paperId":       paper.paperId,
            "title":         paper.title,
            "abstract":      paper.abstract,
            "year":          paper.year,
            "venue":         paper.venue,
            "citationCount": paper.citationCount,
            "authors":       [{"authorId": a.authorId, "name": a.name}
                              for a in (paper.authors or [])],
            "externalIds":   dict(paper.externalIds) if paper.externalIds else {},
            "openAccessPdf": dict(paper.openAccessPdf) if paper.openAccessPdf else None,
        })
        seen_ids.add(paper.paperId)
        collected += 1

    print(f"  ✓ {collected} papers collected for {year}  "
          f"(running total: {len(all_papers)})")

    # Checkpoint save after each year — crash-safe
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_papers, f, indent=2)
    print(f"  ↳ Saved to {OUTPUT_FILE}")

    time.sleep(2)  # small courtesy pause between years

print(f"\n{'='*55}")
print(f"  Done. Total papers: {len(all_papers)}")
print(f"  Saved to: {OUTPUT_FILE}")
print(f"{'='*55}")