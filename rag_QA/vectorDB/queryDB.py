"""
queryDB.py
----------
Ad-hoc script for manually querying the Qdrant vector database.

Embeds a hardcoded query string using SPECTER 2 and prints the top matching
papers with their title, year, authors, citation count, and abstract snippet.
Useful for sanity-checking retrieval quality during development.
"""

from helpers import load_embedding_model, embed_document, get_qdrant_client, query_vector_db

COLLECTION_NAME = "nlp_papers"

# ── Load embedding model ──────────────────────────────────────────────────────

tokenizer, model = load_embedding_model()

# ── Connect to Qdrant ─────────────────────────────────────────────────────────

client = get_qdrant_client()

# ── Query ─────────────────────────────────────────────────────────────────────

user_query = "Improve efficiency of attention mechanisms in transformer models"
print(f"\nSearching for papers similar to: '{user_query}'...\n")

# Embed the query
query_embedding = embed_document([user_query], tokenizer, model)[0]

# Query the vector database
# defaults to top_k=3
results = query_vector_db(client, query_embedding, collection_name=COLLECTION_NAME)

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
    print(f"Abstract: {paper['abstract'][:200]}...")
    print("-" * 80)
