from rag_QA.langgraph.contradiction.nli import detect_contradictions

from pipeline.state import RAGState
from pipeline.constants import NLI_MODEL


def detect_contradictions_node(state: RAGState) -> dict:
    """LangGraph node: detects contradictions among retrieved chunks."""
    contradiction_pairs = detect_contradictions(state["retrieved_chunks"], NLI_MODEL)

    if contradiction_pairs:
        print(f"[nli] found {len(contradiction_pairs)} contradiction(s): {contradiction_pairs}")
    else:
        print("[nli] no contradictions found")

    return {"contradiction_pairs": contradiction_pairs}