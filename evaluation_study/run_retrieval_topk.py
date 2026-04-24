from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.bm25_index import BM25CorpusIndex
from rag_QA.vectorDB.dense_search import dense_search


DATA_DIR = ROOT / "evaluation_study" / "data"
SC1_PATH = DATA_DIR / "sc1_output.parquet"
SC2_PATH = DATA_DIR / "sc2_output.parquet"
SHARED_CORPUS_PATH = DATA_DIR / "shared_corpus.json"


def extract_question_column(df: pd.DataFrame, parquet_name: str) -> str:
    candidates = ["question", "prompt", "query", "input"]
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find a question column in {parquet_name}. "
        f"Tried: {candidates}. Actual columns: {list(df.columns)}"
    )


def retrieval_lists(results: list[dict[str, Any]]) -> tuple[list[str], list[str], list[str], list[float]]:
    doc_ids = [str(r.get("doc_id", "") or "") for r in results]
    titles = [str(r.get("title", "") or "") for r in results]
    texts = [str(r.get("text", "") or "") for r in results]
    scores = [float(r.get("score", 0.0) or 0.0) for r in results]
    return doc_ids, titles, texts, scores


def process_parquet(
    path: Path,
    retriever_name: str,
    retriever_fn: Callable[[str, int], list[dict[str, Any]]],
    top_k: int,
    overwrite: bool = False,
    limit: int | None = None,
) -> None:
    print(f"\n[{retriever_name}] Loading {path}")
    df = pd.read_parquet(path)

    question_col = extract_question_column(df, path.name)
    print(f"[{retriever_name}] Using question column: {question_col}")

    doc_ids_col = f"{retriever_name}_topk_doc_ids"
    titles_col = f"{retriever_name}_topk_titles"
    texts_col = f"{retriever_name}_topk_texts"
    scores_col = f"{retriever_name}_topk_scores"

    target_cols = [doc_ids_col, titles_col, texts_col, scores_col]
    if all(col in df.columns for col in target_cols) and not overwrite:
        print(f"[{retriever_name}] Columns already exist in {path.name}. Skipping.")
        return

    all_doc_ids: list[list[str]] = []
    all_titles: list[list[str]] = []
    all_texts: list[list[str]] = []
    all_scores: list[list[float]] = []

    total = len(df) if limit is None else min(limit, len(df))
    print(f"[{retriever_name}] Processing {total} rows from {path.name}")

    for i, (_, row) in enumerate(df.iloc[:total].iterrows(), start=1):
        query = str(row[question_col])

        print(f"[{retriever_name}] Row {i}/{total}")
        try:
            results = retriever_fn(query, top_k)
        except Exception as e:
            print(f"[{retriever_name}] ERROR on row {i}: {e}")
            results = []

        doc_ids, titles, texts, scores = retrieval_lists(results)
        all_doc_ids.append(doc_ids)
        all_titles.append(titles)
        all_texts.append(texts)
        all_scores.append(scores)

    if limit is not None and limit < len(df):
        remain = len(df) - limit
        all_doc_ids.extend([[] for _ in range(remain)])
        all_titles.extend([[] for _ in range(remain)])
        all_texts.extend([[] for _ in range(remain)])
        all_scores.extend([[] for _ in range(remain)])

    df[doc_ids_col] = all_doc_ids
    df[titles_col] = all_titles
    df[texts_col] = all_texts
    df[scores_col] = all_scores

    df.to_parquet(path, index=False)
    print(f"[{retriever_name}] Saved updated parquet: {path}")
    print(f"[{retriever_name}] Added columns: {target_cols}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retriever",
        choices=["bm25", "dense", "both"],
        default="both",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--files",
        nargs="*",
        choices=["sc1", "sc2"],
        default=["sc1", "sc2"],
    )
    args = parser.parse_args()

    bm25_index = BM25CorpusIndex.from_json(SHARED_CORPUS_PATH)

    def bm25_runner(query: str, top_k: int) -> list[dict[str, Any]]:
        return bm25_index.search(query, top_k=top_k)

    def dense_runner(query: str, top_k: int) -> list[dict[str, Any]]:
        return dense_search(query, top_k=top_k)

    selected_paths: list[Path] = []
    if "sc1" in args.files:
        selected_paths.append(SC1_PATH)
    if "sc2" in args.files:
        selected_paths.append(SC2_PATH)

    for path in selected_paths:
        if args.retriever in {"bm25", "both"}:
            process_parquet(
                path=path,
                retriever_name="bm25",
                retriever_fn=bm25_runner,
                top_k=args.top_k,
                overwrite=args.overwrite,
                limit=args.limit,
            )

        if args.retriever in {"dense", "both"}:
            process_parquet(
                path=path,
                retriever_name="dense",
                retriever_fn=dense_runner,
                top_k=args.top_k,
                overwrite=args.overwrite,
                limit=args.limit,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()