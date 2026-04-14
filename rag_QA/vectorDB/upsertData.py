import json
import time
import os
import uuid
from qdrant_client.models import Distance, VectorParams, PointStruct

from helpers import load_embedding_model, embed, get_qdrant_client

# ── Config ────────────────────────────────────────────────────────────────────

PAPERS_FILE     = "data/papers.json"
COLLECTION_NAME = "nlp_papers"
EMBEDDING_DIM   = 768          # SPECTER 2 output size
BATCH_SIZE      = 25
WAIT_SECONDS    = 20

# ── Load papers ───────────────────────────────────────────────────────────────

print("Loading papers from", PAPERS_FILE)
with open(PAPERS_FILE, "r") as f:
    papers = json.load(f)

# Drop papers with no abstract — can't embed them
papers = [p for p in papers if p.get("abstract")]
print(f"  → {len(papers)} papers with abstracts found\n")

# ── Load embedding model ──────────────────────────────────────────────────────

tokenizer, model = load_embedding_model()

# ── Connect to Qdrant ─────────────────────────────────────────────────────────

client = get_qdrant_client()

# Create the collection if it doesn't exist yet
existing = [c.name for c in client.get_collections().collections]
if COLLECTION_NAME not in existing:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print("  → Collection created\n")
else:
    print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation\n")

# ── Embed and upload in batches ───────────────────────────────────────────────

total_batches = (len(papers) + BATCH_SIZE - 1) // BATCH_SIZE  # ceiling division

for batch_index in range(total_batches):
    start = batch_index * BATCH_SIZE
    end   = start + BATCH_SIZE
    batch = papers[start:end]

    print(f"Batch {batch_index + 1}/{total_batches}  (papers {start + 1}–{min(end, len(papers))})")

    # --- Step 1: embed all abstracts in this batch ---
    abstracts = [paper["abstract"] for paper in batch]
    print(f"  Embedding {len(abstracts)} abstracts...")
    texts = [paper["title"] + tokenizer.sep_token + paper["abstract"] for paper in batch]
    embeddings = embed(texts, tokenizer, model)

    # --- Step 2: build Qdrant points ---
    points = []
    for paper, embedding in zip(batch, embeddings):
        payload = {
            "paperId":       paper.get("paperId"),
            "title":         paper.get("title"),
            "abstract":      paper.get("abstract"),
            "year":          paper.get("year"),
            "venue":         paper.get("venue"),
            "citationCount": paper.get("citationCount"),
            "authors":       paper.get("authors", []),
            "externalIds":   paper.get("externalIds", {}),
            "openAccessPdf": paper.get("openAccessPdf", {}),
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),   # random UUID as the point ID
                vector=embedding.tolist(),
                payload=payload,
            )
        )

    # --- Step 3: upsert to Qdrant ---
    print(f"  Uploading to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  ✓ Batch {batch_index + 1} uploaded successfully")

    # --- Step 4: wait before next batch (skip wait after the last batch) ---
    if batch_index < total_batches - 1:
        print(f"  Waiting {WAIT_SECONDS}s before next batch...\n")
        time.sleep(WAIT_SECONDS)

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\nAll done! {len(papers)} papers uploaded to collection '{COLLECTION_NAME}'.")