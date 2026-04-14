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

# ===================== 5. 生成最终回答（需要拼接数据结构） =====================
# def generate_answer_node(state: RAGState) -> dict:
#     """用检索到的上下文 + 重写后的查询，生成最终回答"""
#     rewritten_query = state["rewritten_query"]
#     retrieved_chunks = state["retrieved_chunks"]

#     # 拼接上下文
#     context = "\n\n".join([f"【上下文 {i+1}】{chunk}" for i, chunk in enumerate(retrieved_chunks)])

#     # 回答提示词
#     answer_prompt = f"""
# 你是一个专业的问答助手，请根据以下提供的上下文，回答用户的问题。
# 要求：
# 1.  只使用上下文里的信息，不要编造
# 2.  回答清晰、准确、有条理
# 3.  如果上下文里没有答案，直接说明“无法根据现有信息回答”

# 上下文：
# {context}

# 用户问题：{rewritten_query}
# 回答：
# """

#     response = llm.invoke(answer_prompt)
#     final_answer = response.content.strip()

#     return {
#         "messages": state["messages"] + [{"role": "assistant", "content": final_answer}]
#     }


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