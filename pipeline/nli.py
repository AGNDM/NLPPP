"""
nli.py
------
LangGraph node wrapping the NLI-based contradiction detector.

Thin adapter between the LangGraph pipeline and the core detect_contradictions()
function in rag_QA/langgraph/contradiction/nli.py. Skips inference entirely if
fewer than two chunks were retrieved, since contradiction detection requires at
least one pair.
"""

from rag_QA.langgraph.contradiction.nli import detect_contradictions

from pipeline.state import RAGState
from pipeline.constants import NLI_MODEL


def detect_contradictions_node(state: RAGState) -> dict[str, list[tuple[int, int]]]:
    """LangGraph node: detects contradictions among retrieved chunks."""
    #Skip contradiction detection if fewer than 2 chunks were retrieved
    if len(state["retrieved_chunks"]) < 2:
        return {"contradiction_pairs": []}
    
    contradiction_pairs = detect_contradictions(state["retrieved_chunks"], NLI_MODEL)

    if contradiction_pairs:
        print(f"[nli] found {len(contradiction_pairs)} contradiction(s): {contradiction_pairs}")
    else:
        print("[nli] no contradictions found")

    return {"contradiction_pairs": contradiction_pairs}
