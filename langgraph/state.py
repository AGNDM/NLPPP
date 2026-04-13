from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class RAGState(TypedDict):
    original_query: str                       # raw query
    rewritten_query: str                    
    retrieved_chunks: list 