# evaluation_study/run_bm25_baseline.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Existing pipeline nodes
from pipeline.rewrite import rewrite_query
from pipeline.grade import grade_chunks
from pipeline.nli import detect_contradictions_node
from pipeline.generate import generate_answer

# BM25 retriever
from pipeline.rag_bm25 import retrieve as bm25_retrieve


DATA_DIR = ROOT / "evaluation_study" / "data"

SC1_PATH = DATA_DIR / "sc1_output.parquet"
SC2_PATH = DATA_DIR / "sc2_output.parquet"

BM25_OUTPUT_COL = "output_checkpoint_bm25"


def build_initial_state(question: str) -> dict[str, Any]:
    return {
        "original_query": question,
        "rewritten_query": question,
        "rewritten_user_question": question,
        "retrieved_chunks": [],
        "contradiction_pairs": [],
        "answer": "",
    }


def extract_question_column(df: pd.DataFrame, parquet_name: str) -> str:
    """
    Find the question/prompt column.
    Adjust this if your parquet uses a different column name.
    """
    candidates = ["question", "prompt", "query", "input"]
    for col in candidates:
        if col in df.columns:
            return col

    raise KeyError(
        f"Could not find a question column in {parquet_name}. "
        f"Tried: {candidates}. Actual columns: {list(df.columns)}"
    )


def run_single_question(question: str) -> str:
    state = build_initial_state(question)

    state.update(rewrite_query(state))
    # bm25 retriever
    state.update(bm25_retrieve(state))
    state.update(grade_chunks(state))
    state.update(detect_contradictions_node(state))
    state.update(generate_answer(state))

    return state["answer"]


def process_parquet(path: Path, overwrite: bool = False, limit: int | None = None) -> None:
    print(f"\n[BM25] Loading {path}")
    df = pd.read_parquet(path)

    question_col = extract_question_column(df, path.name)
    print(f"[BM25] Using question column: {question_col}")

    if BM25_OUTPUT_COL in df.columns and not overwrite:
        print(f"[BM25] Column '{BM25_OUTPUT_COL}' already exists in {path.name}. Skipping.")
        return

    outputs: list[str] = []

    total = len(df) if limit is None else min(limit, len(df))
    print(f"[BM25] Processing {total} rows from {path.name}")

    for i, (_, row) in enumerate(df.iloc[:total].iterrows(), start=1):
        question = str(row[question_col])

        print(f"\n[BM25] Row {i}/{total}")
        print(f"[BM25] Question: {question}")

        try:
            answer = run_single_question(question)
        except Exception as e:
            print(f"[BM25] ERROR on row {i}: {e}")
            answer = ""

        outputs.append(answer)

    # If using limit, keep remaining rows
    if limit is not None and limit < len(df):
        remainder = [""] * (len(df) - limit)
        outputs.extend(remainder)

    df[BM25_OUTPUT_COL] = outputs

    output_path = path
    df.to_parquet(output_path, index=False)
    print(f"[BM25] Saved updated parquet with column '{BM25_OUTPUT_COL}' to: {output_path}")


def main() -> None:
    overwrite = "--overwrite" in sys.argv

    # for smoke test use.
    limit = None
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        try:
            limit = int(sys.argv[idx + 1])
        except (IndexError, ValueError):
            raise ValueError("Usage: --limit <int>")

    process_parquet(SC1_PATH, overwrite=overwrite, limit=limit)
    process_parquet(SC2_PATH, overwrite=overwrite, limit=limit)

    print("\n[BM25] Done.")


if __name__ == "__main__":
    main()