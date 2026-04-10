import os
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from qdrant_client import QdrantClient

from helpers import QDRANT_API_KEY, load_embedding_model, embed, get_qdrant_client, query_vector_db

COLLECTION_NAME = "nlp_papers"

# ── Load embedding model ──────────────────────────────────────────────────────

tokenizer, model = load_embedding_model()

# ── Connect to Qdrant ─────────────────────────────────────────────────────────

client = get_qdrant_client()

# ── Query ─────────────────────────────────────────────────────────────────────

user_query = "Improve efficiency of attention mechanisms in transformer models"
print(f"\nSearching for papers similar to: '{user_query}'...\n")

# Embed the query
query_embedding = embed([user_query], tokenizer, model)[0] # Note how we turn the user query into a list of one item

# Query the vector database
results = query_vector_db(client, query_embedding, collection_name=COLLECTION_NAME) # defaults to top_k=3

# ── Print results ─────────────────────────────────────────────────────────────

print("=" * 80)
for i, hit in enumerate(results, 1):
    paper = hit.payload
    print(f"\n#{i} (Similarity: {hit.score:.3f})")
    print(f"Title: {paper['title']}")
    print(f"Year: {paper['year']}")
    authors = paper.get('authors', [])[:3]
    author_names = [a['name'] if isinstance(a, dict) else a for a in authors]
    print(f"Authors: {', '.join(author_names)}")
    print(f"Citation Count: {paper['citationCount']}")
    print(f"Abstract: {paper['abstract'][:200]}...")  # First 200 chars
    print("-" * 80)