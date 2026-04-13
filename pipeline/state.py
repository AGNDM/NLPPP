from typing import TypedDict

class RAGState(TypedDict):
    original_query: str                       # raw query
    rewritten_query: str                    
    retrieved_chunks:list
    nli_pairs: list