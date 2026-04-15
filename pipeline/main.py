from langgraph.graph import StateGraph, START, END
from pipeline.rewrite import query_rewrite
from pipeline.state import RAGState
from pipeline.rag import query_vector_trunks
from pipeline.nli import nli_detect
from pipeline.generate import generate_answer_node

def restart(state: RAGState):
    return "wait_user_input"

def wait_user_input(state: RAGState):
    user_input = input("\n👉 please input your question: ")
    return {
        "original_query": user_input
    }


# ===================== 6.  LangGraph workflow =====================
graph = StateGraph(RAGState)

# add nodes
graph.add_node("wait_user_input", wait_user_input)
graph.add_node("query_rewrite", query_rewrite)
graph.add_node("get_retrived_trunks", query_vector_trunks)
graph.add_node("contradict_detect", nli_detect)
graph.add_node("generate_answer", generate_answer_node)

# START → rewrite → retrive → NLI → generate answer → END
graph.add_edge(START, "wait_user_input")
graph.add_edge("wait_user_input", "query_rewrite")
graph.add_edge("query_rewrite", "get_retrived_trunks")
graph.add_edge("get_retrived_trunks", "contradict_detect")
graph.add_edge("get_retrived_trunks", "generate_answer")
# graph.add_edge("retrieve", "generate_answer")
# graph.add_conditional_edges("contradict_detect", restart)

# compile
app = graph.compile()


# ===================== 7. test =====================
if __name__ == "__main__":
    pass
    # user input
    # user_query = input("\n👉 please input your question: ")

    # initial state
    initial_state = {
        "original_query": "",
        "rewritten_query": "",
        "retrieved_chunks": [],
        "nli_pairs": []
    }

    # # run workflow
    result = app.invoke(initial_state)

    print("\n🎉 final result：", result)
    # print(result["messages"][-1]["content"])