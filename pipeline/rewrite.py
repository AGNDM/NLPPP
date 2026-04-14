from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from pipeline.state import RAGState
from pipeline.constant import REWRITE_MODEL_NAME

load_dotenv()  # load .env 

# ===================== 1. env setup =====================
OraAgent = ChatOpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=os.getenv("OPENAI_API_KEY"),
  model=REWRITE_MODEL_NAME, 
  temperature=0
)

def query_rewrite(state: RAGState) -> dict:
    """
    This node rewrites the user question into a clean, retrieval-optimised query with Organisation Agent.
    This is the LangChain 'Extend a method' credit item: query rewriting as a prompting strategy beyond zero/few-shot.
    """
    original_query = state["original_query"]

    rewrite_prompt = f"""
You are a professional retrieval optimization assistant, responsible for rewriting user questions into queries more suitable for vector database retrieval.
Requirements:
Remove colloquial, redundant, and irrelevant expressions while retaining the core retrieval intent.
Supplement necessary context to make the query more specific and accurate.
Return only the rewritten query without any explanations, prefixes, or suffixes.
Maintain the same language style as the original question (English).
Original query: {original_query}
Rewritten query:
"""

    # call Organisation Agent
    response = OraAgent.invoke(rewrite_prompt)
    rewritten_query = response.content.strip()

    print(f"\n✅ raw query：{original_query}")
    print(f"✅ rewritten query：{rewritten_query}")

    # update state
    return {
        "rewritten_query": rewritten_query
    }
