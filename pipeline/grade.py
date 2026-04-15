from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

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

Reply with a single word: YES (for relevant) or NO (for not relevant). Nothing else.

Question: {question}

Abstract: {abstract}

Answer:"""


def grade_chunks(state: RAGState) -> dict:
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