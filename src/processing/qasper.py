from models import QASPERExample, ProcessedExample
from typing import Optional, Any
import duckdb
from pathlib import Path
from dotenv import load_dotenv
import os
import pandas as pd


def to_processed(raw: QASPERExample) -> Optional[ProcessedExample]:
    """
    Transforms a raw QASPER row into a ProcessedExample. 
    
    Filters out rows that are unanswerable or lack supporting evidence to 
    ensure high-quality training/evaluation data.
    """
    if raw.is_unanswerable:
        return None

    if raw.evidence == []:
        return None
    
    return ProcessedExample(
        question=raw.question_text,
        abstract=raw.abstract,
        context="\n\n".join(raw.evidence),
        answer = raw.raw_answer,
        source=raw.qasper_id  # ArXiv ID
    )


def get_raw_qasper_examples(data_path: Path) -> tuple[list[str], list[tuple[Any, ...]]]:
    """
    Reads raw QASPER Parquet files using DuckDB and instantiates them 
    into a list of validated Pydantic models.
    """
    path_pattern = str(data_path / "raw")

    con = duckdb.connect()

    query = """FROM read_parquet(?)"""

    cursor = con.execute(query, [path_pattern])
    cols = [desc[0] for desc in con.description]

    examples = [
        QASPERExample(**dict(zip(cols, row)))
        for row in cursor.fetchall()
    ]

    con.close()
    return examples


def process_qasper(data_path: Path):
    """
    Orchestrates the QASPER data pipeline: loads raw Parquet files, 
    filters/transforms the examples, and saves the result.
    """
    processed_dir = data_path/"processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    filename = processed_dir/"processed.parquet"
    
    qasper_examples = get_raw_qasper_examples(data_path)

    processed =  [
        p for e in qasper_examples
        if (p := to_processed(e))
    ]
    pd.DataFrame([p.model_dump() for p in processed]).to_parquet(filename)
    print(f"Written {len(processed)} examples to {filename}")


if __name__ == "__main__":
    _ = load_dotenv()
    process_qasper(Path(os.environ["QASPER_DIR"]))
