from models import QASPERExample, QASData
from datasets import load_dataset, Dataset
import pandas as pd
from dotenv import load_dotenv
import os
from pathlib import Path


def process_qasper_dataset(dataset_dict: dict) -> dict[str, list[QASPERExample]]:
    qasper_dict = {}
    for split_key, split in dataset_dict.items():
        qasper_dict[split_key] = process_split(split, split_key)
    return qasper_dict


def process_split(split: Dataset, split_key: str) -> list[QASPERExample]:
    split_examples = []
    for row in split:
        split_examples.extend(process_row(row, split_key))
    return split_examples 


def process_row(row: dict, split_key: str) -> list[QASPERExample]:
    paper_id = row["id"]
    abstract = row["abstract"]
    column = row["qas"]
    qas_items = process_qas_column(column)

    return [
        QASPERExample(
            qasper_id=paper_id,
            abstract=abstract,
            split=split_key,
            **qas.model_dump()
        )
        for qas in qas_items
    ]


def process_qas_column(column: dict) -> list[QASData]:
    examples = []
    for i, question_id in enumerate(column["question_id"]):
        question_text = column["question"][i]
        nlp_background = column["nlp_background"][i]
        topic_background = column["topic_background"][i]
        paper_read = column["paper_read"][i]
        paper_read = bool(paper_read) if paper_read != "" else None
        search_query = column["search_query"][i]
        examples.append(
            QASData(
                question_id=question_id,
                question_text=question_text,
                nlp_background=nlp_background,
                topic_background=topic_background,
                paper_read=paper_read,
                search_query=search_query,
                **process_answer(column["answers"][i])
            )
        )
    return examples


def process_answer(answers: dict) -> dict:
    annotator_answers = answers["answer"]
    annotator_count = len(annotator_answers)

    votes = [a["unanswerable"] for a in annotator_answers]
    annotators_agree_answersability = len(set(votes)) == 1
    
    is_unanswerable = sum(votes) > annotator_count / 2

    if is_unanswerable:
        return {
            "is_unanswerable": True,
            "annotators_agree_answersability": annotators_agree_answersability,
            "annotator_count": annotator_count,
            "answer_type": None,
            "raw_answer": "",
            "evidence": [],
            "has_float_evidence": False,
            "is_cleanable": False,
        }
    
    candidates = [a for a in annotator_answers if not a["unanswerable"]]
    selected = None
    for candidate in candidates:
        if candidate["free_form_answer"]:
            selected = candidate
            break
        elif candidate["extractive_spans"] and selected is None:
            selected = candidate
        elif candidate["yes_no"] is not None and selected is None:
            selected = candidate

    if selected["free_form_answer"]:
        answer_type = "free_form"
        raw_answer = selected["free_form_answer"]
    elif selected["extractive_spans"]:
        answer_type = "extractive"
        raw_answer = " ".join(selected["extractive_spans"])
    else:
        answer_type = "yes_no"
        raw_answer = "Yes" if selected["yes_no"] else "No"

    all_evidence = selected["evidence"]
    has_float_evidence = any(e.startswith("FLOAT SELECTED") for e in all_evidence)
    evidence = [e for e in all_evidence if not e.startswith("FLOAT SELECTED")]
    
    return {
        "is_unanswerable": False,
        "annotators_agree_answersability": annotators_agree_answersability,
        "annotator_count": annotator_count,
        "answer_type": answer_type,
        "raw_answer": raw_answer,
        "evidence": evidence,
        "has_float_evidence": has_float_evidence,
        "is_cleanable": len(evidence) > 0,
    }


def create_raw_qasper_splits(
        qasper_dir: Path,
        hf_path: str ="allenai/qasper",
        hf_revision: str="refs/pr/6"   #required to bypass deprecated data script

    ):

    
    qasper_raw_dir =  qasper_dir / "raw"
    qasper_raw_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(hf_path, revision=hf_revision)
    qasper_dict = process_qasper_dataset(ds)

    for split, examples in qasper_dict.items():
        filename = qasper_raw_dir/f"{split}.parquet"
        df = pd.DataFrame([example.model_dump() for example in examples])
        df.to_parquet(filename)
        print(f"Written {len(examples)} examples to {filename}")


if __name__ == "__main__":
    _ = load_dotenv()
    create_raw_qasper_splits(qasper_dir=Path(os.environ["QASPER_DIR"]))
