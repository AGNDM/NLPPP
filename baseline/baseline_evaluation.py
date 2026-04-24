"""
bm25_eval.py  —  BM25 vs. SPECTER2 Embedding Retrieval Comparison
===================================================================
For each query, retrieves the top-1 abstract from both methods and
prints them side-by-side so two annotators can independently mark
each result as Relevant (1) or Not Relevant (0).

Usage:
    python bm25_eval.py

Requirements:
    pip install rank_bm25
    (all other deps already in the project environment)
"""

import json
import logging
import sys
import textwrap
from datetime import datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag_QA.vectorDB.helpers import load_query_model, embed_query, get_qdrant_client, query_vector_db


# ── Logging setup ────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    """Create a logger that writes to both stdout and a timestamped log file
    in the same directory as this script."""
    log_path = Path(__file__).parent / f"bm25_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("bm25_eval")
    logger.setLevel(logging.INFO)

    # File handler — writes everything to the log file
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Stream handler — mirrors output to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    print(f"Logging results to: {log_path}\n")
    return logger

# ── Configuration ─────────────────────────────────────────────────────────────

PAPERS_PATH = Path("data/papers.json")
TOP_K = 1  # We retrieve 1 result per method per query for manual evaluation

QUERIES = [
    "How do convolutional neural networks extract events from text?",
    "Can simple bag-of-words deep learning models compete with syntactic models for text classification?",
    "How can recurrent neural networks be used for semantic role labeling without parsing?",
    "Do multi-sense word embeddings improve performance on NLP tasks like POS tagging and sentiment analysis?",
    "How can LSTM networks model long documents for text classification?",
    "How is calibration of probabilistic NLP models evaluated?",
    "How do attention-based bidirectional LSTM networks perform on relation classification?",
]

# ── Load papers from local JSON ───────────────────────────────────────────────

def load_papers(path: Path) -> list[dict]:
    with open(path, "r") as f:
        papers = json.load(f)
    # Drop any papers with missing abstracts
    papers = [p for p in papers if p.get("abstract")]
    print(f"Loaded {len(papers)} papers with abstracts from {path}\n")
    return papers

# ── BM25 retrieval ────────────────────────────────────────────────────────────

def build_bm25_index(papers: list[dict]) -> BM25Okapi:
    """Tokenise each abstract by whitespace (lowercased) and build BM25 index."""
    tokenised_corpus = [p["abstract"].lower().split() for p in papers]
    return BM25Okapi(tokenised_corpus)


def bm25_retrieve(query: str, bm25: BM25Okapi, papers: list[dict], top_k: int) -> list[dict]:
    """Return the top_k papers ranked by BM25 score for the given query."""
    tokenised_query = query.lower().split()
    scores = bm25.get_scores(tokenised_query)
    # argsort ascending, then take last top_k in reverse order
    top_indices = scores.argsort()[-top_k:][::-1]
    return [papers[i] for i in top_indices]

# ── Embedding retrieval ───────────────────────────────────────────────────────

def embedding_retrieve(query: str, tokenizer, model, qdrant_client, top_k: int) -> list[dict]:
    """Embed the query and retrieve top_k results from Qdrant."""
    query_vec = embed_query(query, tokenizer, model)
    results = query_vector_db(qdrant_client, query_vec, top_k=top_k)
    # Convert Qdrant ScoredPoint objects to plain dicts matching papers.json schema
    return [
        {
            "title": r.payload.get("title", "N/A"),
            "abstract": r.payload.get("abstract", "N/A"),
            "score": round(r.score, 4),
        }
        for r in results
    ]

# ── Pretty printer ────────────────────────────────────────────────────────────

def print_results(query_idx: int, query: str, bm25_result: dict, emb_result: dict, logger: logging.Logger):
    width = 80
    sep = "═" * width

    logger.info(f"\n{sep}")
    logger.info(f"QUERY {query_idx}: {query}")
    logger.info(sep)

    # BM25 result
    logger.info("\n  ── BM25 Result ──────────────────────────────────────────────")
    logger.info(f"  Title   : {bm25_result['title']}")
    logger.info(f"  Abstract:")
    for line in textwrap.wrap(bm25_result["abstract"], width=74):
        logger.info(f"    {line}")

    logger.info(f"\n  [ Annotator 1 relevance: ___ ]   [ Annotator 2 relevance: ___ ]")

    # Embedding result
    logger.info("\n  ── Embedding (SPECTER2) Result ──────────────────────────────")
    logger.info(f"  Title   : {emb_result['title']}")
    logger.info(f"  Score   : {emb_result.get('score', 'N/A')}")
    logger.info(f"  Abstract:")
    for line in textwrap.wrap(emb_result["abstract"], width=74):
        logger.info(f"    {line}")

    logger.info(f"\n  [ Annotator 1 relevance: ___ ]   [ Annotator 2 relevance: ___ ]")
    logger.info("")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 0. Set up logger (stdout + log file)
    logger = setup_logger()

    # 1. Load papers from disk
    papers = load_papers(PAPERS_PATH)

    # 2. Build BM25 index over abstract strings
    logger.info("Building BM25 index...")
    bm25 = build_bm25_index(papers)
    logger.info("  → BM25 index ready\n")

    # 3. Load SPECTER2 query model + Qdrant client
    tokenizer, model = load_query_model()
    qdrant_client = get_qdrant_client()

    # 4. Run both retrievers for every query and print/log results
    for i, query in enumerate(QUERIES, start=1):
        bm25_top = bm25_retrieve(query, bm25, papers, top_k=TOP_K)
        emb_top  = embedding_retrieve(query, tokenizer, model, qdrant_client, top_k=TOP_K)

        print_results(
            query_idx   = i,
            query       = query,
            bm25_result = bm25_top[0],
            emb_result  = emb_top[0],
            logger      = logger,
        )

    # 5. Log blank scoring summary table for annotators to fill in
    logger.info("═" * 80)
    logger.info("SCORING SUMMARY  —  fill in 1 (relevant) or 0 (not relevant)")
    logger.info("═" * 80)
    header = f"{'Query':<6} {'BM25 A1':<10} {'BM25 A2':<10} {'Emb A1':<10} {'Emb A2':<10}"
    logger.info(header)
    logger.info("-" * 46)
    for i in range(1, len(QUERIES) + 1):
        logger.info(f"Q{i:<5} {'___':<10} {'___':<10} {'___':<10} {'___':<10}")
    logger.info("")
    logger.info("Precision@1 per method = (sum of relevant marks) / number of queries")
    logger.info("Inter-annotator agreement: compare A1 vs A2 columns per method")


if __name__ == "__main__":
    main()