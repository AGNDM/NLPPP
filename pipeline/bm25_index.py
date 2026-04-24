from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


class BM25CorpusIndex:
    """
    In-memory BM25 index over the shared corpus.

    Expected shared corpus schema per document:
    {
        "doc_id": str,
        "title": str,
        "abstract": str,
        "text": str,       # title + abstract
        "year": int | None,
        "venue": str
    }
    """

    def __init__(self, corpus_records: list[dict[str, Any]]) -> None:
        self.records = corpus_records
        self.tokenized_corpus = [_tokenize(rec.get("text", "")) for rec in corpus_records]
        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_json(cls, path: str | Path) -> "BM25CorpusIndex":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}")

        return cls(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for idx in ranked_indices:
            rec = self.records[idx]
            results.append(
                {
                    "doc_id": rec.get("doc_id", ""),
                    "title": rec.get("title", ""),
                    "text": rec.get("text", ""),
                    "abstract": rec.get("abstract", ""),
                    "year": rec.get("year"),
                    "venue": rec.get("venue", ""),
                    "score": float(scores[idx]),
                }
            )

        return results