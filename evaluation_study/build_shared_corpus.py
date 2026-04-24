from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_INPUT = ROOT / "rag_QA"/"vectorDB"/"data" / "papers.json"
DEFAULT_OUTPUT = ROOT / "evaluation_study" / "data" / "shared_corpus.json"


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_doc_record(paper: dict[str, Any]) -> dict[str, Any]:
    """
    Build one shared-corpus document record.

    Shared retrieval text:
        title + " " + abstract

    Minimal required fields:
        doc_id, title, abstract, text

    Optional metadata:
        year, venue
    """
    doc_id = safe_str(paper.get("paperId"))
    title = safe_str(paper.get("title"))
    abstract = safe_str(paper.get("abstract"))
    year = paper.get("year")
    venue = safe_str(paper.get("venue"))

    text = f"{title} {abstract}".strip()

    return {
        "doc_id": doc_id,
        "title": title,
        "abstract": abstract,
        "text": text,
        "year": year,
        "venue": venue,
    }


def is_valid_paper(paper: dict[str, Any]) -> bool:
    """
    Keep only papers that have:
    - a non-empty paperId
    - a non-empty abstract
    """
    paper_id = safe_str(paper.get("paperId"))
    abstract = safe_str(paper.get("abstract"))
    return bool(paper_id and abstract)


def deduplicate_by_doc_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Deduplicate corpus records by doc_id, keeping the first occurrence.
    """
    seen = set()
    deduped = []

    for record in records:
        doc_id = record["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        deduped.append(record)

    return deduped


def build_shared_corpus(input_path: Path) -> list[dict[str, Any]]:
    print(f"Loading papers from: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    if not isinstance(papers, list):
        raise ValueError("Expected papers.json to contain a JSON list.")

    print(f"Loaded {len(papers)} raw papers")

    valid_papers = [paper for paper in papers if is_valid_paper(paper)]
    print(f"Kept {len(valid_papers)} papers with non-empty paperId and abstract")

    records = [build_doc_record(paper) for paper in valid_papers]
    records = deduplicate_by_doc_id(records)

    print(f"Final shared corpus size after deduplication: {len(records)}")
    return records


def save_json(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved shared corpus to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    # Path to papers.json
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    # Path to save shared corpus JSON
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()

    records = build_shared_corpus(args.input)
    save_json(records, args.output)


if __name__ == "__main__":
    main()