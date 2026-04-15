from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from pipeline.state import RAGState
from pipeline.constants import REWRITE_MODEL

load_dotenv()

# Initialised once at import time so the node doesn't recreate it on every call.
# temperature=0 keeps rewrites deterministic.
_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=REWRITE_MODEL,
    temperature=0,
)

_PROMPT = _PROMPT = """\
You are a retrieval optimisation assistant. Given a user question, produce two outputs:

1. RETRIEVAL QUERY
   A short, keyword-dense phrase for semantic search over NLP research paper abstracts.
   Use technical terminology and noun phrases only — no full sentences, no verbs, no filler words.
   Example: "multihead attention mechanism transformer self-attention"

2. REWRITTEN QUESTION
   A clean, detailed, self-contained version of the user's question suitable for an LLM to answer.
   Keep it as a natural question but remove all colloquialisms and vagueness.
   Add relevant technical context if it is clearly implied by the original question.
   Example: "How does the multihead attention mechanism work in transformer architectures, \
and what is the role of the individual attention heads?"

Return your answer in this exact format and nothing else by replacing the palce holders with the actual retrieval query and rewritten question:
<retrieval_query> | <rewritten_question>

Question: {question}
Answer:"""


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the user's raw question into a retrieval-optimised query."""
    response = _llm.invoke(_PROMPT.format(question=state["original_query"]))
    rewritten = response.content.strip()
    print("rewritten query and user question:", rewritten)
    question_parts = rewritten.split("|")
    if len(question_parts) != 2:
        raise ValueError(f"Unexpected rewrite format: '{rewritten}'")
    rewritten_query, rewritten_user_question = question_parts

    print(f"[rewrite] rewritten_query: '{rewritten_query}'")
    print(f"[rewrite] rewritten_user_question: '{rewritten_user_question}'")

    return {
        "rewritten_query": rewritten_query,
        "rewritten_user_question": rewritten_user_question
    }