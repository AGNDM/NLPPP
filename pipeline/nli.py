from pipeline.state import RAGState
from rag_QA.langgraph.contradiction.nli import detect_contradictions
from pipeline.constant import NLI_MODLE_NAME


def nli_detect(state: RAGState):
    nli_pairs = detect_contradictions(state["retrieved_chunks"], NLI_MODLE_NAME)
    return {
        "nli_pairs": nli_pairs
    }