from pipeline.state import RAGState
from rag_QA.langgraph.contradiction.nli import detect_contradictions

def nli_detect(state: RAGState):
    nli_pairs = detect_contradictions(state["retrieved_chunks"], "cross-encoder/nli-deberta-v3-large")
    return {
        "nli_pairs": nli_pairs
    }