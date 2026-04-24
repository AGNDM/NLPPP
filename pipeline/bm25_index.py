"""
bm25.py
-------
Standalone BM25 index over a pre-built document corpus.

Provides BM25CorpusIndex, a self-contained in-memory BM25 index that can be
constructed directly from a list of records or loaded from a JSON file. Used
in the ablation study to provide a lexical retrieval baseline comparable to
the dense SPECTER 2 retrieval pipeline.

The expected corpus schema is documented in BM25CorpusIndex.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    """Lowercase and tokenize text into word tokens for BM25 indexing."""
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
        """
        Build a BM25 index from a list of corpus records.

        Args:
            corpus_records: List of document dicts following the corpus schema.
                            Each record must contain a 'text' field for indexing.
        """
        self.records = corpus_records
        self.tokenized_corpus = [_tokenize(rec.get("text", "")) for rec in corpus_records]
        self.index = BM25Okapi(self.tokenized_corpus)

    @classmethod
    def from_json(cls, path: str | Path) -> "BM25CorpusIndex":
        """
        Load a corpus from a JSON file and build a BM25 index over it.

        Args:
            path: Path to a JSON file containing a list of corpus records.

        Returns:
            A BM25CorpusIndex instance built from the loaded records.

        Raises:
            ValueError: If the JSON file does not contain a list.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected a JSON list in {path}")

        return cls(records)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Retrieve the top-k documents most relevant to a query using BM25.

        Args:
            query:  The search query string.
            top_k:  Number of results to return. Defaults to 5.

        Returns:
            List of dicts following the corpus schema, each augmented with a
            'score' field containing the BM25 score. Returns an empty list if
            the query tokenizes to nothing.
        """
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
