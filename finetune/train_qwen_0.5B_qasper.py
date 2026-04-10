#!/usr/bin/env python3
"""Finetune Qwen-0.5B on the Qasper dataset.

- Dataset: Parquet file at `data/qasper/processed/tinetuning_dataset.parquet`
- Columns: `prompt` (input) and `response` (target)
- Uses HuggingFace `transformers` Trainer.
- Intended to be launched with `accelerate launch` for multi‑GPU (4× GH200) support.
- After training, generates 5 sample completions and writes them to `samples.txt`.
"""

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen-0.5B on Qasper")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/qasper/processed/tinetuning_dataset.parquet",
        help="Path to the parquet dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen-0.5B",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="finetune/output_qwen_0.5B",
        help="Directory to save checkpoints and final model",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per GPU",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation to reach effective batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--samples_output",
        type=str,
        default="samples.txt",
        help="File to write generated samples",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of sample generations after training",
    )
    return parser.parse_args()


def load_dataset(data_path: str) -> Dataset:
    df = pd.read_parquet(data_path)
    # Expect columns `prompt` and `response`
    if not {"prompt", "response"}.issubset(df.columns):
        raise ValueError("Parquet file must contain 'prompt' and 'response' columns")
    # Combine prompt and response into a single text for causal LM training
    df["text"] = df["prompt"].astype(str) + "\n" + df["response"].astype(str)
    return Dataset.from_pandas(df[["text"]])


def tokenize_function(examples, tokenizer, max_len):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer and model with bfloat16 for GH200
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load and tokenize dataset
    raw_dataset = load_dataset(args.data_path)
    tokenized_dataset = raw_dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
        batched=True,
        remove_columns=["text"],
    )

    # Data collator for causal LM (labels are inputs shifted)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("=== Starting training ===")
    trainer.train()
    print("=== Training completed, saving model ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # ---- Sample generation ----
    print(f"Generating {args.num_samples} sample completions...")
    # Use first N prompts from the original parquet for demonstration
    df = pd.read_parquet(args.data_path)
    sample_prompts = df["prompt"].astype(str).head(args.num_samples).tolist()
    model.eval()
    generated_texts = []
    for i, prompt in enumerate(sample_prompts):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(f"--- Sample {i+1} ---\nPrompt: {prompt}\nGenerated: {gen_text}\n")

    samples_path = Path(args.samples_output)
    samples_path.write_text("\n".join(generated_texts))
    print(f"Samples written to {samples_path.resolve()}")

if __name__ == "__main__":
    main()
