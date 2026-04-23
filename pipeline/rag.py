"""
rag.py
------
LangGraph node for embedding and retrieval.

Embeds the rewritten query using SPECTER 2's adhoc_query adapter and retrieves
the top-K most similar paper chunks from Qdrant above a similarity threshold.
Vectors are included in the results for downstream cosine similarity
pre-filtering in the NLI contradiction detection step.

The SPECTER 2 query model and Qdrant client are initialised once at import
time — both are expensive to load and should not be recreated per query.
"""

from dotenv import load_dotenv
from qdrant_client.models import ScoredPoint

from rag_QA.vectorDB.helpers import load_query_model, embed_query, get_qdrant_client

from pipeline.state import RAGState
from pipeline.constants import (
    RETRIEVAL_COLLECTION,
    RETRIEVAL_TOP_K,
)

load_dotenv()

# Initialised once at import time — loading SPECTER2 is expensive and
# should not happen on every query.
_query_tokenizer, _query_model = load_query_model()

_qdrant = get_qdrant_client()


def retrieve(state: RAGState) -> dict[str, list[ScoredPoint]]:
    """Embed the rewritten query and fetch the top-K most similar papers from Qdrant."""
    query_vector = embed_query(state["rewritten_query"], _query_tokenizer, _query_model).tolist()

    results = _qdrant.query_points(
        collection_name=RETRIEVAL_COLLECTION,
        query=query_vector,
        score_threshold=0.8,
        limit=RETRIEVAL_TOP_K,
        with_payload=True,
        with_vectors=True,  # needed downstream for NLI cosine similarity pre-filter
    ).points

    # print each retrieved chunk's title and similarity score for debugging
    print(f"[retrieve] retrieved {len(results)} chunks:")
    for i, chunk in enumerate(results):
        title = chunk.payload.get("title", "No title")
        score = chunk.score
        print(f"  {i + 1}. {title} (score: {score:.4f})")

    print(f"[retrieve] found {len(results)} chunks for query: '{state['rewritten_query']}'")

    return {"retrieved_chunks": results}
