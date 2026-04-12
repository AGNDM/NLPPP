#!/usr/bin/env python3
"""Fine-tune Llama 8B on a Tulu-style instruction dataset."""

import argparse
import importlib.util
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	Trainer,
	TrainingArguments,
)

try:
	from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
	LoraConfig = None
	TaskType = None
	get_peft_model = None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fine-tune Llama 8B on a Tulu dataset")
	parser.add_argument(
		"--model_name",
		type=str,
		default="meta-llama/Meta-Llama-3-8B",
		help="Hugging Face model ID for the base model",
	)
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="allenai/tulu-3-sft-mixture",
		help="HF dataset name for Tulu (used when --data_path is not provided)",
	)
	parser.add_argument(
		"--data_path",
		type=str,
		default="",
		help="Local dataset path (.json/.jsonl/.parquet). Overrides --dataset_name.",
	)
	parser.add_argument("--train_split", type=str, default="train")
	parser.add_argument("--output_dir", type=str, default="finetune/output_llama_tulu_8B")

	parser.add_argument("--max_seq_length", type=int, default=2048)
	parser.add_argument("--max_train_samples", type=int, default=None)

	parser.add_argument("--learning_rate", type=float, default=2e-5)
	parser.add_argument("--num_train_epochs", type=float, default=2.0)
	parser.add_argument("--per_device_train_batch_size", type=int, default=1)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--weight_decay", type=float, default=0.01)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=200)
	parser.add_argument("--save_total_limit", type=int, default=2)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument(
		"--lora_target_modules",
		type=str,
		nargs="+",
		default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	)

	return parser.parse_args()


def _has_accelerate() -> bool:
	return importlib.util.find_spec("accelerate") is not None


def _normalize_role(role: str) -> str:
	role_lower = str(role).strip().lower()
	if role_lower in {"user", "assistant", "system"}:
		return role_lower
	if role_lower in {"human", "instruction"}:
		return "user"
	if role_lower in {"bot", "model", "response", "output"}:
		return "assistant"
	return "user"


def _to_messages(record: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
	for key in ("messages", "conversation", "conversations"):
		if key in record and isinstance(record[key], list):
			normalized: List[Dict[str, str]] = []
			for turn in record[key]:
				if not isinstance(turn, dict):
					continue
				role = _normalize_role(turn.get("role", turn.get("from", "user")))
				content = turn.get("content", turn.get("value", turn.get("text", "")))
				content = str(content).strip()
				if content:
					normalized.append({"role": role, "content": content})
			if normalized:
				return normalized

	return None


def _load_local_dataset(path: str, split: str) -> Dataset:
	lower = path.lower()
	if lower.endswith(".parquet"):
		frame = pd.read_parquet(path)
		return Dataset.from_pandas(frame)
	if lower.endswith(".json"):
		return load_dataset("json", data_files=path, split="train")
	if lower.endswith(".jsonl"):
		return load_dataset("json", data_files=path, split="train")
	raise ValueError("Unsupported --data_path. Use .json, .jsonl, or .parquet")


def _render_chat_text(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
	chat_template = getattr(tokenizer, "chat_template", None)
	if chat_template:
		return tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=False,
		)

	# Fallback for base models without chat_template:
	# keep Tulu role/content turns and render with Llama-3 header/eot style.
	parts: List[str] = ["<|begin_of_text|>"]
	for turn in messages:
		role = str(turn.get("role", "user")).strip().lower()
		content = str(turn.get("content", "")).strip()
		if not content:
			continue
		if role not in {"system", "user", "assistant"}:
			role = "user"
		parts.append(f"<|start_header_id|>{role}<|end_header_id|>\\n\\n{content}<|eot_id|>")

	return "".join(parts)


def _render_turn_text(role: str, content: str) -> str:
	role = str(role).strip().lower()
	if role not in {"system", "user", "assistant"}:
		role = "user"
	return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content.strip()}<|eot_id|>"


def build_text_dataset(raw_dataset: Dataset, tokenizer: AutoTokenizer) -> Dataset:
	rows: List[Dict[str, Any]] = []
	for record in raw_dataset:
		messages = _to_messages(record)
		if not messages:
			continue

		text = _render_chat_text(messages, tokenizer)
		if text:
			rows.append({"messages": messages, "text": text})

	if not rows:
		raise ValueError("No trainable examples found in the provided dataset")

	return Dataset.from_list(rows)


def tokenize_function(examples: Dict[str, List[Any]], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, Any]:
	input_ids_batch: List[List[int]] = []
	attention_mask_batch: List[List[int]] = []
	labels_batch: List[List[int]] = []

	bos_token_ids = tokenizer("<|begin_of_text|>", add_special_tokens=False)["input_ids"]
	if not bos_token_ids:
		bos_token_ids = []

	for messages in examples["messages"]:
		full_ids: List[int] = list(bos_token_ids)
		full_labels: List[int] = [-100] * len(bos_token_ids)

		for turn in messages:
			if not isinstance(turn, dict):
				continue
			role = _normalize_role(turn.get("role", turn.get("from", "user")))
			content = str(turn.get("content", turn.get("value", turn.get("text", "")))).strip()
			if not content:
				continue

			segment_text = _render_turn_text(role, content)
			segment_ids = tokenizer(segment_text, add_special_tokens=False)["input_ids"]
			full_ids.extend(segment_ids)
			if role == "assistant":
				full_labels.extend(segment_ids)
			else:
				full_labels.extend([-100] * len(segment_ids))

		full_ids = full_ids[:max_length]
		full_labels = full_labels[:max_length]
		attention_mask = [1] * len(full_ids)

		input_ids_batch.append(full_ids)
		attention_mask_batch.append(attention_mask)
		labels_batch.append(full_labels)

	return {
		"input_ids": input_ids_batch,
		"attention_mask": attention_mask_batch,
		"labels": labels_batch,
	}


def collate_causal_lm(features: List[Dict[str, List[int]]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
	max_len = max(len(feature["input_ids"]) for feature in features)
	pad_id = tokenizer.pad_token_id
	if pad_id is None:
		raise ValueError("Tokenizer must have a pad_token_id")

	input_ids_batch: List[List[int]] = []
	attention_mask_batch: List[List[int]] = []
	labels_batch: List[List[int]] = []

	for feature in features:
		input_ids = list(feature["input_ids"])
		attention_mask = list(feature["attention_mask"])
		labels = list(feature["labels"])
		pad_len = max_len - len(input_ids)
		if pad_len < 0:
			input_ids = input_ids[:max_len]
			attention_mask = attention_mask[:max_len]
			labels = labels[:max_len]
			pad_len = 0

		input_ids_batch.append(input_ids + [pad_id] * pad_len)
		attention_mask_batch.append(attention_mask + [0] * pad_len)
		labels_batch.append(labels + [-100] * pad_len)

	return {
		"input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
		"attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
		"labels": torch.tensor(labels_batch, dtype=torch.long),
	}


def main() -> None:
	args = parse_args()

	if not _has_accelerate():
		raise RuntimeError("Missing dependency: accelerate. Install with `pip install accelerate`.")
	if get_peft_model is None or LoraConfig is None or TaskType is None:
		raise RuntimeError("Missing dependency: peft. Install with `pip install peft`.")

	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	if args.data_path:
		raw_dataset = _load_local_dataset(args.data_path, args.train_split)
	else:
		raw_dataset = load_dataset(args.dataset_name, split=args.train_split)

	if args.max_train_samples is not None:
		raw_dataset = raw_dataset.select(range(min(args.max_train_samples, len(raw_dataset))))

	text_dataset = build_text_dataset(raw_dataset, tokenizer)
	tokenized_dataset = text_dataset.map(
		lambda batch: tokenize_function(batch, tokenizer, args.max_seq_length),
		batched=True,
		remove_columns=["messages", "text"],
	)

	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		torch_dtype=torch.bfloat16,
	)
	model.config.pad_token_id = tokenizer.pad_token_id
	model.config.use_cache = False

	lora_config = LoraConfig(
		r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		target_modules=args.lora_target_modules,
		task_type=TaskType.CAUSAL_LM,
		bias="none",
	)
	model = get_peft_model(model, lora_config)
	if hasattr(model, "enable_input_require_grads"):
		model.enable_input_require_grads()
	else:
		def _set_require_grad(_module, _input, output):
			output.requires_grad_(True)
		model.get_input_embeddings().register_forward_hook(_set_require_grad)

	trainable_param_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
	if trainable_param_count == 0:
		raise RuntimeError(
			"No trainable parameters found after applying LoRA. "
			"Please verify --lora_target_modules matches your model architecture."
		)
	model.print_trainable_parameters()

	training_args = TrainingArguments(
		output_dir=args.output_dir,
		learning_rate=args.learning_rate,
		weight_decay=args.weight_decay,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		warmup_ratio=args.warmup_ratio,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		save_strategy="steps",
		bf16=True,
		fp16=False,
		gradient_checkpointing=False,
		report_to="none",
		seed=args.seed,
		ddp_find_unused_parameters=False,
	)

	data_collator = lambda features: collate_causal_lm(features, tokenizer)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset,
		processing_class=tokenizer,
		data_collator=data_collator,
	)

	trainer.train()
	trainer.save_model(args.output_dir)
	tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
	main()
