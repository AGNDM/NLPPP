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

_PROMPT = """\
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

IMPORTANT: Your entire response must be exactly one line in this format and nothing else:
<retrieval_query> | <rewritten_question>

Do not write any explanation, preamble, or extra text. Only the single line above.

Question: {question}
Answer:"""


def _parse(response_text: str) -> tuple[str, str] | None:
    """Return (retrieval_query, rewritten_question) if the format is valid, else None."""
    parts = response_text.strip().split("|")
    if len(parts) == 2 and all(p.strip() for p in parts):
        return parts[0].strip(), parts[1].strip()
    return None


def rewrite_query(state: RAGState) -> dict:
    """Rewrite the user's raw question into a retrieval-optimised query.

    Retries once if the model ignores the format instruction.
    Falls back to the original query if both attempts fail.
    """
    question = state["original_query"]

    for attempt in range(2):
        response = _llm.invoke(_PROMPT.format(question=question))
        raw = response.content.strip()
        print(f"[rewrite] attempt {attempt + 1} raw output: '{raw}'")

        parsed = _parse(raw)
        if parsed:
            rewritten_query, rewritten_user_question = parsed
            print(f"[rewrite] rewritten_query:         '{rewritten_query}'")
            print(f"[rewrite] rewritten_user_question: '{rewritten_user_question}'")
            return {
                "rewritten_query": rewritten_query,
                "rewritten_user_question": rewritten_user_question,
            }

        print(f"[rewrite] attempt {attempt + 1} failed to match expected format, retrying...")

    # Both attempts failed — fall back to the original question for both fields
    # so the pipeline can still run rather than crashing.
    print(f"[rewrite] WARNING: falling back to original query after 2 failed attempts")
    return {
        "rewritten_query": question,
        "rewritten_user_question": question,
    }