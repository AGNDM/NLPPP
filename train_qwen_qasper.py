#!/usr/bin/env python3
import argparse
import importlib.util
import inspect
import json
import os
import random
import tarfile
import tempfile
import urllib.request
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    prepare_model_for_kbit_training = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a small Qwen model on allenai/qasper.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./outputs/qwen-qasper-lora")
    parser.add_argument("--dataset_name", type=str, default="allenai/qasper")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="validation")

    parser.add_argument("--max_doc_chars", type=int, default=5000)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

    parser.add_argument("--use_4bit", action="store_true")

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=float, default=2.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_text_chunks(node: Any) -> List[str]:
    if node is None:
        return []
    if isinstance(node, str):
        text = node.strip()
        return [text] if text else []
    if isinstance(node, (int, float)):
        return [str(node)]
    if isinstance(node, list):
        chunks: List[str] = []
        for item in node:
            chunks.extend(_extract_text_chunks(item))
        return chunks
    if isinstance(node, dict):
        chunks: List[str] = []
        for value in node.values():
            chunks.extend(_extract_text_chunks(value))
        return chunks
    return []


def _build_context(record: Dict[str, Any], max_doc_chars: int) -> str:
    title = " ".join(_extract_text_chunks(record.get("title", "")))
    abstract = " ".join(_extract_text_chunks(record.get("abstract", "")))
    full_text = " ".join(_extract_text_chunks(record.get("full_text", "")))

    context_parts = []
    if title:
        context_parts.append(f"Title: {title}")
    if abstract:
        context_parts.append(f"Abstract: {abstract}")
    if full_text:
        context_parts.append(f"Paper: {full_text}")

    context = "\n".join(context_parts).strip()
    return context[:max_doc_chars]


def _normalize_one_answer(answer_dict: Dict[str, Any]) -> Optional[str]:
    if not isinstance(answer_dict, dict):
        return None

    answer_payload = answer_dict.get("answer", answer_dict)

    free_form = " ".join(_extract_text_chunks(answer_payload.get("free_form_answer", ""))) if isinstance(answer_payload, dict) else ""
    if free_form.strip():
        return free_form.strip()

    yes_no = answer_payload.get("yes_no") if isinstance(answer_payload, dict) else None
    if isinstance(yes_no, bool):
        return "Yes" if yes_no else "No"

    extractive = " ".join(_extract_text_chunks(answer_payload.get("extractive_spans", ""))) if isinstance(answer_payload, dict) else ""
    if extractive.strip():
        return extractive.strip()

    unanswerable = answer_payload.get("unanswerable") if isinstance(answer_payload, dict) else None
    if isinstance(unanswerable, bool) and unanswerable:
        return "Unanswerable"

    return None


def _choose_answer(answers: Any) -> Optional[str]:
    if not isinstance(answers, list):
        return None
    for candidate in answers:
        text = _normalize_one_answer(candidate)
        if text:
            return text
    return None


def build_sft_examples(split_ds: Dataset, max_doc_chars: int) -> Dataset:
    rows: List[Dict[str, str]] = []

    for record in split_ds:
        context = _build_context(record, max_doc_chars=max_doc_chars)
        qas = record.get("qas", [])
        if not isinstance(qas, list):
            continue

        for qa_item in qas:
            if not isinstance(qa_item, dict):
                continue
            question = " ".join(_extract_text_chunks(qa_item.get("question", ""))).strip()
            answer = _choose_answer(qa_item.get("answers", []))

            if not question or not answer or not context:
                continue

            text = (
                "You are a helpful scientific QA assistant. "
                "Answer the question based only on the provided paper content.\n\n"
                f"### Paper Context\n{context}\n\n"
                f"### Question\n{question}\n\n"
                f"### Answer\n{answer}"
            )
            rows.append({"text": text})

    return Dataset.from_list(rows)


def tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
    tokenized = tokenizer(examples["text"], truncation=True, max_length=max_length, padding=False)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def _has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def _build_training_arguments(args: argparse.Namespace, has_eval: bool, torch_dtype: torch.dtype) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "bf16": torch_dtype == torch.bfloat16,
        "fp16": torch_dtype == torch.float16,
        "gradient_checkpointing": True,
        "report_to": "none",
        "seed": args.seed,
    }

    strategy_value = "steps" if has_eval else "no"

    init_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in init_params:
        kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in init_params:
        kwargs["eval_strategy"] = strategy_value

    supported_kwargs = {key: value for key, value in kwargs.items() if key in init_params}
    return TrainingArguments(**supported_kwargs)


def _build_trainer(
    model: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset],
    data_collator: Any,
    tokenizer: Any,
) -> Trainer:
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_init_params:
        kwargs["tokenizer"] = tokenizer
    elif "processing_class" in trainer_init_params:
        kwargs["processing_class"] = tokenizer

    supported_kwargs = {key: value for key, value in kwargs.items() if key in trainer_init_params}
    return Trainer(**supported_kwargs)


def _records_from_qasper_json_bytes(blob: bytes) -> List[Dict[str, Any]]:
    payload = json.loads(blob.decode("utf-8"))
    rows: List[Dict[str, Any]] = []
    for article_id, article in payload.items():
        if isinstance(article, dict):
            item = dict(article)
            item["id"] = article_id
            rows.append(item)
    return rows


def _extract_qasper_json_from_tgz(tgz_path: str, target_name: str) -> List[Dict[str, Any]]:
    with tarfile.open(tgz_path, "r:gz") as archive:
        for member in archive.getmembers():
            if member.name.endswith(target_name):
                extracted = archive.extractfile(member)
                if extracted is None:
                    break
                return _records_from_qasper_json_bytes(extracted.read())
    raise FileNotFoundError(f"Cannot find {target_name} in archive: {tgz_path}")


def load_qasper_with_fallback(dataset_name: str) -> Dict[str, Dataset]:
    try:
        loaded = load_dataset(dataset_name)
        return {split: loaded[split] for split in loaded.keys()}
    except RuntimeError as err:
        message = str(err)
        if "Dataset scripts are no longer supported" not in message or dataset_name != "allenai/qasper":
            raise

    print("Detected datasets script loading restriction. Falling back to manual QASPER download...")
    train_dev_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
    test_url = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

    with tempfile.TemporaryDirectory(prefix="qasper_") as tmp_dir:
        train_dev_tgz = os.path.join(tmp_dir, "qasper-train-dev-v0.3.tgz")
        test_tgz = os.path.join(tmp_dir, "qasper-test-and-evaluator-v0.3.tgz")

        urllib.request.urlretrieve(train_dev_url, train_dev_tgz)
        urllib.request.urlretrieve(test_url, test_tgz)

        train_rows = _extract_qasper_json_from_tgz(train_dev_tgz, "qasper-train-v0.3.json")
        dev_rows = _extract_qasper_json_from_tgz(train_dev_tgz, "qasper-dev-v0.3.json")
        test_rows = _extract_qasper_json_from_tgz(test_tgz, "qasper-test-v0.3.json")

    return {
        "train": Dataset.from_list(train_rows),
        "validation": Dataset.from_list(dev_rows),
        "test": Dataset.from_list(test_rows),
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print(f"Loading dataset: {args.dataset_name}")
    raw = load_qasper_with_fallback(args.dataset_name)

    train_raw = raw[args.train_split]
    eval_raw = raw[args.eval_split] if args.eval_split in raw else None

    if args.max_train_samples is not None:
        train_raw = train_raw.select(range(min(len(train_raw), args.max_train_samples)))
    if eval_raw is not None and args.max_eval_samples is not None:
        eval_raw = eval_raw.select(range(min(len(eval_raw), args.max_eval_samples)))

    print("Building SFT datasets from QASPER...")
    train_ds = build_sft_examples(train_raw, max_doc_chars=args.max_doc_chars)
    eval_ds = build_sft_examples(eval_raw, max_doc_chars=args.max_doc_chars) if eval_raw is not None else None

    if len(train_ds) == 0:
        raise ValueError("No train samples generated. Please check preprocessing logic or dataset format.")

    print(f"Train examples: {len(train_ds)}")
    if eval_ds is not None:
        print(f"Eval examples: {len(eval_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    quantization_config = None
    if args.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    print(f"Loading model: {args.model_name_or_path}")
    from_pretrained_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch_dtype,
        "quantization_config": quantization_config,
    }
    if args.use_4bit:
        if _has_accelerate():
            from_pretrained_kwargs["device_map"] = "auto"
        else:
            print("accelerate is not installed; loading without device_map. Install accelerate for better memory placement.")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **from_pretrained_kwargs,
    )

    if args.use_lora:
        if LoraConfig is None or get_peft_model is None:
            raise ImportError("PEFT is not installed. Please install peft to use --use_lora.")
        if args.use_4bit and prepare_model_for_kbit_training is not None:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    train_tokenized = train_ds.map(
        lambda batch: tokenize_function(batch, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )

    eval_tokenized = None
    if eval_ds is not None and len(eval_ds) > 0:
        eval_tokenized = eval_ds.map(
            lambda batch: tokenize_function(batch, tokenizer=tokenizer, max_length=args.max_length),
            batched=True,
            remove_columns=eval_ds.column_names,
            desc="Tokenizing eval set",
        )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = _build_training_arguments(
        args=args,
        has_eval=eval_tokenized is not None,
        torch_dtype=torch_dtype,
    )

    trainer = _build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Start training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
