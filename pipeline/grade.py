"""
grade.py
--------
LangGraph node for relevance grading of retrieved paper chunks.

After retrieval from Qdrant, not all returned chunks are guaranteed to be
directly useful for answering the question. This module filters them by
prompting an LLM to judge whether each abstract contains facts or findings
that would concretely help answer the query — domain similarity alone is
not sufficient to pass.

Reuses the same OpenRouter LLM instance as the query rewriter (see rewrite.py).
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from qdrant_client.models import ScoredPoint

from pipeline.state import RAGState
from pipeline.constants import REWRITE_MODEL

load_dotenv()

# Reuse the same OpenRouter LLM as the query rewriter.
_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model=REWRITE_MODEL,
    temperature=0,
)

_PROMPT = """\
You are a strict relevance grader for a QA system about NLP research.

Your job is to decide whether the abstract below contains information that \
directly and specifically helps answer the question. \
Domain similarity is NOT enough — the abstract must contain facts, findings, \
or details that the downstream answering model can actually use to answer the question.
When grading, lean towards marking borderline cases as "relevant" to avoid filtering out potentially useful information, 
but do not mark clearly irrelevant abstracts as relevant.

Reply with a single word: YES (for relevant) or NO (for not relevant). Nothing else.

Question: {question}

Abstract: {abstract}

Answer:"""


def grade_chunks(state: RAGState) -> dict[str, list[ScoredPoint]]:
    """LangGraph node: filters retrieved chunks to only those directly relevant to the query."""
    question = state["rewritten_user_question"]
    chunks = state["retrieved_chunks"]

    relevant_chunks = []
    for chunk in chunks:
        abstract = chunk.payload.get("abstract", "")
        prompt = _PROMPT.format(question=question, abstract=abstract)
        response = _llm.invoke(prompt).content.strip().upper()

        title = chunk.payload.get("title", "No title")
        if response.startswith("YES"):
            print(f"[grade] RELEVANT: {title}")
            relevant_chunks.append(chunk)
        else:
            print(f"[grade] FILTERED: {title}")

    print(f"[grade] {len(relevant_chunks)}/{len(chunks)} chunks passed the relevance filter")
    return {"retrieved_chunks": relevant_chunks}
