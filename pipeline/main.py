from langgraph.graph import StateGraph, START, END

from pipeline.state import RAGState
from pipeline.rewrite import rewrite_query
from pipeline.rag import retrieve
from pipeline.nli import detect_contradictions_node
from pipeline.generate import generate_answer
from pipeline.grade import grade_chunks


def ask_user(state: RAGState) -> dict:
    """Entry node — reads the user's question from stdin."""
    question = input("\n\n ACTION REQUIRED\nEnter your question: ").strip()
    return {"original_query": question}


# ── Graph definition ──────────────────────────────────────────────────────────

graph = StateGraph(RAGState)

graph.add_node("ask_user", ask_user)
graph.add_node("rewrite_query", rewrite_query)
graph.add_node("retrieve", retrieve)
graph.add_node("grade_chunks", grade_chunks)
graph.add_node("detect_contradictions", detect_contradictions_node)
graph.add_node("generate_answer", generate_answer)
 
# Linear pipeline: each step feeds into the next
graph.add_edge(START, "ask_user")
graph.add_edge("ask_user", "rewrite_query")
graph.add_edge("rewrite_query", "retrieve")
graph.add_edge("retrieve", "grade_chunks")
graph.add_edge("grade_chunks", "detect_contradictions")
graph.add_edge("detect_contradictions", "generate_answer")
graph.add_edge("generate_answer", END)

app = graph.compile()

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    initial_state: RAGState = {
        "original_query": "",
        "rewritten_query": "",
        "rewritten_user_question": "",
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }

    result = app.invoke(initial_state)

    print("\n── Answer ───────────────────────────────────────────────")
    print(result["answer"])