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