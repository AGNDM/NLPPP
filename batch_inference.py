from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import List

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA checkpoint inference across multiple GPUs.")
    parser.add_argument("--input_parquet", type=str, default="eval_sc1_inference.parquet", help="Input parquet file or directory")
    parser.add_argument("--output_parquet", type=str, default="batch_inference_results.parquet", help="Output parquet file")
    parser.add_argument("--model_name", type=str, default="allenai/Llama-3.1-Tulu-3-8B", help="Base model name")
    parser.add_argument("--adapter_root", type=str, default="tulu_qasper_lora_output", help="Directory containing checkpoint-* folders")
    parser.add_argument("--start_checkpoint", type=int, default=10, help="First checkpoint number")
    parser.add_argument("--end_checkpoint", type=int, default=130, help="Last checkpoint number (inclusive)")
    parser.add_argument("--checkpoint_step", type=int, default=10, help="Checkpoint step size")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--batch_size", type=int, default=4, help="Generation batch size per GPU")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 = greedy")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling top-p")
    return parser.parse_args()


def _discover_checkpoints(adapter_root: str, start: int, end: int, step: int) -> List[Path]:
    root = Path(adapter_root)
    checkpoints = []
    for checkpoint in range(start, end + 1, step):
        path = root / f"checkpoint-{checkpoint}"
        if path.exists():
            checkpoints.append(path)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {root}")
    return checkpoints


def _chunk_round_robin(items: List[Path], num_chunks: int) -> List[List[Path]]:
    return [items[i::num_chunks] for i in range(num_chunks)]


def _prepare_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_base_model(model_name: str):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = True
    return model


def _generate_for_checkpoint(
    model,
    tokenizer,
    df: pd.DataFrame,
    checkpoint_path: Path,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    lora_model = PeftModel.from_pretrained(model, str(checkpoint_path))
    lora_model.eval()

    do_sample = temperature > 0
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    generation_kwargs = {key: value for key, value in generation_kwargs.items() if value is not None}

    prompts = df["input"].astype(str).tolist()
    outputs: List[str] = []

    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            device = next(lora_model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_len = inputs["input_ids"].shape[1]

            generated_ids = lora_model.generate(**inputs, **generation_kwargs)
            batch_texts = [
                tokenizer.decode(generated_ids[i, prompt_len:], skip_special_tokens=True).strip()
                for i in range(generated_ids.shape[0])
            ]
            outputs.extend(batch_texts)

    del lora_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs


def _worker(
    gpu_id: int,
    checkpoint_paths: List[Path],
    input_parquet: str,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    output_path: str,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    df = pd.read_parquet(input_parquet).reset_index(drop=True)
    tokenizer = _prepare_tokenizer(model_name)
    model = _load_base_model(model_name)

    partial = pd.DataFrame({"row_id": df.index})
    for checkpoint_path in checkpoint_paths:
        checkpoint_num = checkpoint_path.name.split("-")[-1]
        print(f"[GPU {gpu_id}] Running checkpoint {checkpoint_num}...")
        outputs = _generate_for_checkpoint(
            model=model,
            tokenizer=tokenizer,
            df=df,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        partial[f"output_checkpoint_{checkpoint_num}"] = outputs

    partial.to_parquet(output_path, index=False)
    print(f"[GPU {gpu_id}] Wrote {output_path}")


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.input_parquet).reset_index(drop=True)

    checkpoint_paths = _discover_checkpoints(
        adapter_root=args.adapter_root,
        start=args.start_checkpoint,
        end=args.end_checkpoint,
        step=args.checkpoint_step,
    )
    checkpoint_chunks = [chunk for chunk in _chunk_round_robin(checkpoint_paths, args.num_gpus) if chunk]

    with tempfile.TemporaryDirectory(prefix="batch_inference_") as tmpdir:
        ctx = mp.get_context("spawn")
        processes = []
        temp_files = []

        for gpu_id, chunk in enumerate(checkpoint_chunks):
            temp_file = os.path.join(tmpdir, f"gpu_{gpu_id}.parquet")
            temp_files.append(temp_file)
            proc = ctx.Process(
                target=_worker,
                args=(
                    gpu_id,
                    chunk,
                    args.input_parquet,
                    args.model_name,
                    args.batch_size,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_p,
                    temp_file,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("One of the GPU workers failed. Check the logs above.")

        merged = df.copy()
        merged.insert(0, "row_id", merged.index)
        for temp_file in temp_files:
            partial = pd.read_parquet(temp_file)
            merged = merged.merge(partial, on="row_id", how="left")

        merged = merged.sort_values("row_id").drop(columns=["row_id"])
        merged.to_parquet(args.output_parquet, index=False)

    print(f"Inference complete. Results saved to {args.output_parquet}")


if __name__ == "__main__":
    main()