from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


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
    parser.add_argument("--debug", action="store_true", help="Enable debug logs")
    parser.add_argument("--debug_samples", type=int, default=2, help="Number of samples to preview in debug logs")
    parser.add_argument("--debug_chars", type=int, default=300, help="Max characters per debug preview")
    parser.add_argument(
        "--disable_chat_template",
        action="store_true",
        help="Do not wrap input prompts with tokenizer chat template",
    )
    parser.add_argument(
        "--base_output_column",
        type=str,
        default="output_base_model",
        help="Column name for base-model (no adapter) outputs",
    )
    parser.add_argument(
        "--skip_base_model",
        action="store_true",
        help="Skip base-model (no adapter) inference column",
    )
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


def _load_model(model_name: str, target_device: int, checkpoint_path: Path | None = None):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": target_device} if torch.cuda.is_available() else None,
    )
    if checkpoint_path is not None:
        model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = True
    return model


def _cleanup_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _truncate_for_log(text: str, max_chars: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _debug_gpu_memory(gpu_id: int, stage: str, debug: bool) -> None:
    if not debug:
        return
    pid = os.getpid()
    if not torch.cuda.is_available():
        print(f"[DEBUG][GPU {gpu_id}][PID {pid}] {stage}: CUDA not available")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(gpu_id)
    allocated_bytes = torch.cuda.memory_allocated(gpu_id)
    reserved_bytes = torch.cuda.memory_reserved(gpu_id)
    print(
        f"[DEBUG][GPU {gpu_id}][PID {pid}] {stage}: "
        f"free={free_bytes / 1024**3:.2f}GiB "
        f"allocated={allocated_bytes / 1024**3:.2f}GiB "
        f"reserved={reserved_bytes / 1024**3:.2f}GiB "
        f"total={total_bytes / 1024**3:.2f}GiB"
    )


def _format_prompts(tokenizer, prompts: List[str], use_chat_template: bool) -> List[str]:
    if not use_chat_template:
        return prompts
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.chat_template is None:
        print("[WARN] tokenizer has no chat_template; falling back to raw input prompts")
        return prompts

    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def _compute_perplexities(
    model,
    tokenizer,
    prompts: List[str],
    outputs: List[str],
    batch_size: int = 4,
) -> List[float]:
    """Compute perplexity for each generated output.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of original prompts
        outputs: List of generated outputs
        batch_size: Batch size for evaluation
        
    Returns:
        List of perplexity scores (one per output)
    """
    perplexities = []
    
    with torch.inference_mode():
        for batch_start in range(0, len(outputs), batch_size):
            batch_end = min(batch_start + batch_size, len(outputs))
            batch_prompts = prompts[batch_start:batch_end]
            batch_outputs = outputs[batch_start:batch_end]
            
            # Concatenate prompt + output for evaluation
            full_texts = [f"{p}{o}" for p, o in zip(batch_prompts, batch_outputs)]
            
            inputs = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            
            # Get prompt token lengths to identify output portion
            prompt_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            prompt_inputs = {key: value.to(device) for key, value in prompt_inputs.items()}
            prompt_lengths = (prompt_inputs["attention_mask"] == 1).sum(dim=1).tolist()
            
            # Forward pass to get logits
            outputs_dict = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            logits = outputs_dict.logits
            
            # Compute loss for each sample
            for i, (prompt_len, full_text) in enumerate(zip(prompt_lengths, full_texts)):
                # Get indices of output tokens (after prompt)
                input_ids = inputs["input_ids"][i]
                
                # Ensure we have tokens for output part
                if prompt_len >= len(input_ids):
                    perplexities.append(float('inf'))
                    continue
                
                # Compute loss for output tokens
                output_logits = logits[i, prompt_len - 1 : -1, :]
                output_ids = input_ids[prompt_len:]
                
                if len(output_ids) == 0:
                    perplexities.append(float('inf'))
                    continue
                
                # Cross entropy loss
                shift_logits = output_logits.contiguous()
                shift_labels = output_ids.contiguous()
                
                # Use CrossEntropyLoss
                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction='mean',
                )
                
                # Perplexity = exp(loss)
                ppl = float(torch.exp(ce_loss).cpu())
                perplexities.append(ppl)
    
    return perplexities


def _generate_with_model(
    model,
    tokenizer,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    debug: bool = False,
    debug_every_batches: int = 10,
    log_prefix: str = "",
    compute_perplexity: bool = True,
) -> tuple[List[str], List[float]]:
    do_sample = temperature > 0
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    outputs: List[str] = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    start_time = time.time()

    if debug:
        print(f"[DEBUG]{log_prefix} generation_start prompts={len(prompts)} batches={total_batches} batch_size={batch_size}")

    with torch.inference_mode():
        for batch_idx, start in enumerate(range(0, len(prompts), batch_size), start=1):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=False,
            )
            device = next(model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
            prompt_width = inputs["input_ids"].shape[1]

            generated_ids = model.generate(**inputs, generation_config=generation_config)
            batch_texts = [
                tokenizer.decode(generated_ids[i, prompt_width:], skip_special_tokens=True).strip()
                for i in range(generated_ids.shape[0])
            ]
            outputs.extend(batch_texts)

            if debug and (batch_idx % max(1, debug_every_batches) == 0 or batch_idx == total_batches):
                elapsed = time.time() - start_time
                avg_batch_sec = elapsed / batch_idx
                print(
                    f"[DEBUG]{log_prefix} generation_progress "
                    f"batch={batch_idx}/{total_batches} elapsed={elapsed:.1f}s avg_batch={avg_batch_sec:.2f}s"
                )

    if debug:
        total_elapsed = time.time() - start_time
        print(f"[DEBUG]{log_prefix} generation_done outputs={len(outputs)} elapsed={total_elapsed:.1f}s")
    
    perplexities = []
    if compute_perplexity:
        if debug:
            print(f"[DEBUG]{log_prefix} computing_perplexities...")
        perplexities = _compute_perplexities(model, tokenizer, prompts, outputs, batch_size=batch_size)
        if debug:
            print(f"[DEBUG]{log_prefix} perplexities_computed mean={sum(perplexities) / len(perplexities):.4f}")
    else:
        perplexities = [0.0] * len(outputs)
    
    return outputs, perplexities


def _worker(
    gpu_id: int,
    checkpoint_paths: List[Path],
    input_parquet: str,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    include_base_model: bool,
    base_output_column: str,
    use_chat_template: bool,
    debug: bool,
    debug_samples: int,
    debug_chars: int,
    output_path: str,
) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
    if debug:
        current_device = torch.cuda.current_device() if torch.cuda.is_available() else -1
        device_name = torch.cuda.get_device_name(current_device) if torch.cuda.is_available() else "cpu"
        print(
            f"[DEBUG][GPU {gpu_id}][PID {os.getpid()}] "
            f"current_device={current_device} device_name={device_name}"
        )
    _debug_gpu_memory(gpu_id, "worker_start", debug)

    df = pd.read_parquet(input_parquet).reset_index(drop=True)
    raw_prompts = df["input"].astype(str).tolist()
    tokenizer = _prepare_tokenizer(model_name)
    prompts = _format_prompts(
        tokenizer=tokenizer,
        prompts=raw_prompts,
        use_chat_template=use_chat_template,
    )

    if debug and gpu_id == 0:
        print(f"[DEBUG][GPU {gpu_id}] input rows={len(df)}, columns={list(df.columns)}")
        print(f"[DEBUG][GPU {gpu_id}] use_chat_template={use_chat_template}, tokenizer_has_chat_template={tokenizer.chat_template is not None}")
        preview_count = max(0, min(debug_samples, len(prompts)))
        for i in range(preview_count):
            print(f"[DEBUG][GPU {gpu_id}] raw_input[{i}]={_truncate_for_log(raw_prompts[i], debug_chars)}")
            print(f"[DEBUG][GPU {gpu_id}] model_prompt[{i}]={_truncate_for_log(prompts[i], debug_chars)}")

    partial = pd.DataFrame({"row_id": df.index})
    if include_base_model and gpu_id == 0:
        print(f"[GPU {gpu_id}] Running base model (no adapter)...")
        _debug_gpu_memory(gpu_id, "before_base_load", debug)
        model = _load_model(model_name, target_device=gpu_id)
        _debug_gpu_memory(gpu_id, "after_base_load", debug)
        base_outputs, base_perplexities = _generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            debug=debug,
            debug_every_batches=5,
            log_prefix=f"[GPU {gpu_id}][base]",
            compute_perplexity=True,
        )
        partial[base_output_column] = base_outputs
        partial[f"{base_output_column}_ppl"] = base_perplexities
        if debug:
            preview_count = max(0, min(debug_samples, len(base_outputs)))
            for i in range(preview_count):
                print(f"[DEBUG][GPU {gpu_id}] {base_output_column}[{i}]={_truncate_for_log(base_outputs[i], debug_chars)}")
                print(f"[DEBUG][GPU {gpu_id}] {base_output_column}_ppl[{i}]={base_perplexities[i]:.4f}")
        _cleanup_model(model)
        _debug_gpu_memory(gpu_id, "after_base_cleanup", debug)

    for checkpoint_path in checkpoint_paths:
        checkpoint_num = checkpoint_path.name.split("-")[-1]
        print(f"[GPU {gpu_id}] Running checkpoint {checkpoint_num}...")
        _debug_gpu_memory(gpu_id, f"before_ckpt_{checkpoint_num}_load", debug)
        model = _load_model(model_name, target_device=gpu_id, checkpoint_path=checkpoint_path)
        _debug_gpu_memory(gpu_id, f"after_ckpt_{checkpoint_num}_load", debug)
        outputs, perplexities = _generate_with_model(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            debug=debug,
            debug_every_batches=10,
            log_prefix=f"[GPU {gpu_id}][ckpt {checkpoint_num}]",
            compute_perplexity=True,
        )
        partial[f"output_checkpoint_{checkpoint_num}"] = outputs
        partial[f"output_checkpoint_{checkpoint_num}_ppl"] = perplexities
        if debug:
            preview_count = max(0, min(debug_samples, len(outputs)))
            for i in range(preview_count):
                print(f"[DEBUG][GPU {gpu_id}] output_checkpoint_{checkpoint_num}[{i}]={_truncate_for_log(outputs[i], debug_chars)}")
                print(f"[DEBUG][GPU {gpu_id}] output_checkpoint_{checkpoint_num}_ppl[{i}]={perplexities[i]:.4f}")
        _cleanup_model(model)
        _debug_gpu_memory(gpu_id, f"after_ckpt_{checkpoint_num}_cleanup", debug)

    partial.to_parquet(output_path, index=False)
    print(f"[GPU {gpu_id}] Wrote {output_path}")


def main() -> None:
    args = parse_args()
    input_table = pq.read_table(args.input_parquet)
    num_rows = input_table.num_rows

    if args.debug:
        print(f"[DEBUG] input parquet: {args.input_parquet}")
        print(f"[DEBUG] input rows: {num_rows}")
        print(f"[DEBUG] input columns: {input_table.column_names}")
        print(f"[DEBUG] input schema: {input_table.schema}")

    checkpoint_paths = _discover_checkpoints(
        adapter_root=args.adapter_root,
        start=args.start_checkpoint,
        end=args.end_checkpoint,
        step=args.checkpoint_step,
    )
    if args.debug:
        checkpoint_names = [path.name for path in checkpoint_paths]
        print(f"[DEBUG] discovered checkpoints ({len(checkpoint_names)}): {checkpoint_names}")

    checkpoint_chunks = [chunk for chunk in _chunk_round_robin(checkpoint_paths, args.num_gpus) if chunk]
    if args.debug:
        chunk_info = {gpu_id: [path.name for path in chunk] for gpu_id, chunk in enumerate(checkpoint_chunks)}
        print(f"[DEBUG] worker checkpoint assignment: {chunk_info}")

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
                    not args.skip_base_model,
                    args.base_output_column,
                    not args.disable_chat_template,
                    args.debug,
                    args.debug_samples,
                    args.debug_chars,
                    temp_file,
                ),
            )
            proc.start()
            processes.append(proc)

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("One of the GPU workers failed. Check the logs above.")

        merged = pd.DataFrame({"row_id": range(num_rows)})
        for temp_file in temp_files:
            partial = pd.read_parquet(temp_file)
            merged = merged.merge(partial, on="row_id", how="left")

        merged = merged.sort_values("row_id").drop(columns=["row_id"]).reset_index(drop=True)

        if args.debug:
            print(f"[DEBUG] merged output columns: {list(merged.columns)}")

        output_table = input_table
        for column_name in merged.columns:
            output_table = output_table.append_column(
                column_name,
                pa.array(merged[column_name].astype(str).tolist(), type=pa.string()),
            )
        pq.write_table(output_table, args.output_parquet)

        if args.debug:
            print(f"[DEBUG] final output columns: {output_table.column_names}")
            print(f"[DEBUG] final output schema: {output_table.schema}")

    print(f"Inference complete. Results saved to {args.output_parquet}")


if __name__ == "__main__":
    main()