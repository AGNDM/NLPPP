"""
dense.py
--------
Dense retrieval over the Qdrant vector index for the ablation study.

Provides dense_search(), which embeds a query using SPECTER 2 and retrieves
the top-k most similar paper chunks from Qdrant. Results are normalised into
a flat dict structure shared with the BM25 retrieval module so that downstream
saving and evaluation logic can treat both retrieval methods identically.

The SPECTER 2 query model and Qdrant client are initialised once at import time.
"""

from __future__ import annotations
from typing import Any
from .helpers import load_query_model, embed_query, get_qdrant_client


COLLECTION_NAME = "nlp_papers"
DEFAULT_TOP_K = 5

# Load once
_query_tokenizer, _query_model = load_query_model()
_qdrant = get_qdrant_client()


def dense_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Dense retrieval over the existing Qdrant index.

    Returns a normalized list of retrieved docs so downstream saving logic
    can be shared with BM25.

    Args:
        query:  The search query string.
        top_k:  Number of results to return. Defaults to DEFAULT_TOP_K.

    Returns:
        List of dicts, each containing:
            doc_id, title, abstract, text (title + abstract),
            year, venue, and score (cosine similarity).
    """
    query_vector = embed_query(query, _query_tokenizer, _query_model).tolist()

    results = _qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    ).points

    normalized: list[dict[str, Any]] = []
    for point in results:
        payload = point.payload or {}
        title = payload.get("title", "") or ""
        abstract = payload.get("abstract", "") or ""

        normalized.append(
            {
                "doc_id": payload.get("paperId", ""),
                "title": title,
                "text": f"{title} {abstract}".strip(),
                "abstract": abstract,
                "year": payload.get("year"),
                "venue": payload.get("venue", "") or "",
                "score": float(getattr(point, "score", 0.0) or 0.0),
            }
        )

    return normalized
