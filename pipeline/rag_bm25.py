# pipeline/rag_bm25.py

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from rag_QA.vectorDB.helpers import get_qdrant_client
from pipeline.state import RAGState
from pipeline.constants import RETRIEVAL_COLLECTION, RETRIEVAL_TOP_K

load_dotenv()


@dataclass
class BM25ScoredChunk:
    """
    Qdrant-like chunk object so downstream nodes can stay unchanged.

    Downstream code currently expects:
    - chunk.payload
    - chunk.score
    - chunk.vector (contradiction module may rely on this)
    """
    id: Any
    payload: dict
    vector: Any
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", (text or "").lower())


def _build_doc_text(payload: dict) -> str:
    title = payload.get("title", "") or ""
    abstract = payload.get("abstract", "") or ""
    authors = payload.get("authors", "") or ""
    venue = payload.get("venue", "") or ""
    year = str(payload.get("year", "") or "")
    # lexical fields used by BM25
    return f"{title} {abstract} {authors} {venue} {year}".strip()


def _load_collection_records(client, collection_name: str) -> list[Any]:
    """
    Load all records from Qdrant once and keep payload/vector available.
    This keeps the corpus identical to the current dense pipeline,
    while changing only the ranking method.
    """
    all_records = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name=collection_name,
            with_payload=True,
            with_vectors=True,
            limit=256,
            offset=offset,
        )
        all_records.extend(records)

        if next_offset is None:
            break
        offset = next_offset

    return all_records


# Initialize once at import time, same pattern as pipeline/rag.py
_qdrant = get_qdrant_client()
_records = _load_collection_records(_qdrant, RETRIEVAL_COLLECTION)

_corpus_tokens: list[list[str]] = []
_doc_texts: list[str] = []

for record in _records:
    payload = record.payload or {}
    doc_text = _build_doc_text(payload)
    _doc_texts.append(doc_text)
    _corpus_tokens.append(_tokenize(doc_text))

_bm25 = BM25Okapi(_corpus_tokens)

print(f"[bm25] indexed {len(_records)} papers from Qdrant collection '{RETRIEVAL_COLLECTION}'")


def retrieve(state: RAGState) -> dict:
    """
    BM25 replacement for the dense retrieve() node.

    Input:
        state["rewritten_query"]

    Output:
        {"retrieved_chunks": list[BM25ScoredChunk]}

    All downstream logic remains unchanged.
    """
    query = state["rewritten_query"]
    tokenized_query = _tokenize(query)

    if not tokenized_query:
        print("[bm25] empty tokenized query; returning no chunks")
        return {"retrieved_chunks": []}

    scores = _bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    top_indices = [i for i in ranked_indices if scores[i] > 0][:RETRIEVAL_TOP_K]

    results: list[BM25ScoredChunk] = []
    for i in top_indices:
        record = _records[i]
        payload = record.payload or {}
        vector = getattr(record, "vector", None)

        results.append(
            BM25ScoredChunk(
                id=getattr(record, "id", i),
                payload=payload,
                vector=vector,
                score=float(scores[i]),
            )
        )

    print(f"[bm25] retrieved {len(results)} chunks:")
    for i, chunk in enumerate(results, start=1):
        title = chunk.payload.get("title", "No title")
        print(f" {i}. {title} (bm25 score: {chunk.score:.4f})")

    print(f"[bm25] found {len(results)} chunks for query: '{query}'")
    return {"retrieved_chunks": results}