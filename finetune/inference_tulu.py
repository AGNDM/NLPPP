#!/usr/bin/env python3
"""Quick inference for a Tulu-style fine-tuned checkpoint.

Supports either:
- a prompt string, which is wrapped as a single user turn
- a JSON file containing a list of chat messages with `role` and `content`

The script automatically loads PEFT LoRA adapters when the checkpoint
directory contains `adapter_config.json`.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a Tulu-style checkpoint")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Base model name used when loading a PEFT checkpoint",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the saved checkpoint or final output directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Single user prompt to test",
    )
    parser.add_argument(
        "--messages_json",
        type=str,
        default="",
        help="Optional JSON file containing a list of chat messages",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system prompt added before a single user prompt",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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


def _render_chat_text(messages: List[Dict[str, str]], tokenizer: AutoTokenizer) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    parts: List[str] = ["<|begin_of_text|>"]
    for turn in messages:
        role = str(turn.get("role", "user")).strip().lower()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        if role not in {"system", "user", "assistant"}:
            role = "user"
        parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")

    parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(parts)


def _truncate_generated_text(text: str) -> str:
    stop_patterns = [
        r"\n\s*user\s*\n",
        r"\n\s*assistant\s*\n",
        r"\n\s*system\s*\n",
        r"<\|eot_id\|>",
        r"<\|start_header_id\|>",
        r"<\|end_header_id\|>",
    ]
    cut_points = [len(text)]
    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            cut_points.append(match.start())
    return text[: min(cut_points)].strip()


def _build_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.messages_json:
        message_path = Path(args.messages_json)
        if not message_path.exists():
            raise FileNotFoundError(f"messages_json not found: {message_path}")
        payload = json.loads(message_path.read_text())
        if not isinstance(payload, list):
            raise ValueError("messages_json must contain a JSON list of chat messages")
        messages: List[Dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            role = _normalize_role(item.get("role", item.get("from", "user")))
            content = str(item.get("content", item.get("value", item.get("text", "")))).strip()
            if content:
                messages.append({"role": role, "content": content})
        if not messages:
            raise ValueError("messages_json did not contain any usable messages")
        return messages

    if not args.prompt:
        raise ValueError("Provide either --prompt or --messages_json")

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt.strip()})
    messages.append({"role": "user", "content": args.prompt.strip()})
    return messages


def _load_model_and_tokenizer(args: argparse.Namespace):
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint_dir not found: {checkpoint_dir}")

    tokenizer_source = checkpoint_dir if (checkpoint_dir / "tokenizer_config.json").exists() else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
    else:
        torch_dtype = torch.float32
        device = torch.device("cpu")

    adapter_config = checkpoint_dir / "adapter_config.json"
    model_source = args.model_name if adapter_config.exists() else checkpoint_dir

    if adapter_config.exists():
        print(
            f"Detected LoRA adapter checkpoint at {checkpoint_dir}. "
            f"Loading base model '{args.model_name}' and applying adapter weights from the checkpoint."
        )
    else:
        print(
            f"No adapter_config.json found in {checkpoint_dir}. "
            "Loading model weights directly from the checkpoint directory."
        )

    load_kwargs: Dict[str, Any] = {"dtype": torch_dtype}
    if torch.cuda.is_available() and _has_accelerate():
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(str(model_source), **load_kwargs)

    if adapter_config.exists():
        if PeftModel is None:
            raise RuntimeError("Missing dependency: peft. Install with `pip install peft`.")
        model = PeftModel.from_pretrained(model, str(checkpoint_dir))
        print("Confirmed: running the finetuned LoRA model, not the base model.")
    else:
        print("Warning: checkpoint does not look like a LoRA adapter checkpoint.")

    if not torch.cuda.is_available() or "device_map" not in load_kwargs:
        model = model.to(device)

    model.eval()
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = True
    return model, tokenizer, device


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model, tokenizer, device = _load_model_and_tokenizer(args)
    messages = _build_messages(args)
    prompt_text = _render_chat_text(messages, tokenizer)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    do_sample = args.temperature > 0
    eos_token_ids = [tokenizer.eos_token_id]
    eot_token_ids = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
    eot_token_id = eot_token_ids[0] if eot_token_ids else None
    if eot_token_id is not None and eot_token_id not in eos_token_ids and eot_token_id != tokenizer.unk_token_id:
        eos_token_ids.append(eot_token_id)
        print(f"Added <|eot_id|> (token ID: {eot_token_id}) to eos_token_ids")

    print(f"EOS token IDs: {eos_token_ids}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=do_sample,
            temperature=args.temperature if do_sample else None,
            top_p=args.top_p if do_sample else None,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos_token_ids if len(eos_token_ids) > 1 else eos_token_ids[0],
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation_ids = output_ids[0][input_ids.shape[1]:]
    generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    generated_text = _truncate_generated_text(generated_text)

    print("=== Prompt ===")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print("\n=== Generated ===")
    print(generated_text)


if __name__ == "__main__":
    main()