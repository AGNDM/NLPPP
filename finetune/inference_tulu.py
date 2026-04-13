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
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search (1=greedy)")
    parser.add_argument("--early_stopping", action="store_true", help="Stop beam search when any beam hits EOS")
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

    # Prompt for assistant turn: format it exactly like training data format
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

    # Configure EOS tokens for stopping generation
    # Try to find it first, then fall back to the tokenizer's eos_token_id.
    eot_token_ids = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
    eot_token_id = None
    
    # Check if convert_tokens_to_ids returned a valid ID (not unk_token_id)
    if eot_token_ids and eot_token_ids[0] != tokenizer.unk_token_id:
        eot_token_id = eot_token_ids[0]
    else:
        # Fallback: try encoding the string directly
        encoded = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        if encoded and encoded[0] != tokenizer.unk_token_id:
            eot_token_id = encoded[0]
    
    # Build eos_token_ids list with <|eot_id|> as primary if found
    eos_token_ids = []
    if eot_token_id is not None:
        eos_token_ids.append(eot_token_id)
        print(f"Using <|eot_id|> (token ID: {eot_token_id}) as primary EOS token")
    
    # Add tokenizer.eos_token_id as fallback if different
    if tokenizer.eos_token_id not in eos_token_ids:
        eos_token_ids.append(tokenizer.eos_token_id)
    
    if not eos_token_ids:
        raise RuntimeError("Could not find any valid EOS tokens")

    print(f"EOS token IDs: {eos_token_ids}")
    print(f"Primary EOS: {eos_token_ids[0]} (will stop when model generates this token)")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            repetition_penalty=args.repetition_penalty,
            eos_token_id=eos_token_ids,
            pad_token_id=tokenizer.pad_token_id,
        )

    continuation_ids = output_ids[0][input_ids.shape[1]:]
    
    # Decode with special_tokens to see if model actually generated EOS
    generated_text_with_special = tokenizer.decode(continuation_ids, skip_special_tokens=False).strip()
    generated_text = tokenizer.decode(continuation_ids, skip_special_tokens=True).strip()
    
    # Debug: check what token stopped the generation
    print("\n=== Generation Debug ===")
    print(f"Total output length: {len(output_ids[0])}")
    print(f"Input length: {input_ids.shape[1]}")
    print(f"Generated tokens: {len(continuation_ids)}")
    
    # Check if any of the generated tokens are in eos_token_ids
    eot_appearances = sum(1 for tid in continuation_ids if tid in eos_token_ids)
    print(f"EOS tokens (128001, 128009) in generated sequence: {eot_appearances}")
    
    if len(continuation_ids) > 0:
        last_token_id = continuation_ids[-1].item() if hasattr(continuation_ids[-1], 'item') else continuation_ids[-1]
        last_token_text = tokenizer.decode([last_token_id])
        print(f"Last generated token ID: {last_token_id} ({repr(last_token_text)})")
        
        # Show last 5 tokens for context
        if len(continuation_ids) >= 5:
            last_5_ids = [tid.item() if hasattr(tid, 'item') else tid for tid in continuation_ids[-5:]]
            print(f"Last 5 token IDs: {last_5_ids}")
            last_5_text = tokenizer.decode(last_5_ids)
            print(f"Last 5 tokens decoded: {repr(last_5_text)}")
        
        if last_token_id in eos_token_ids:
            print(f"✓ Stopped at EOS token (as expected)")
        elif len(continuation_ids) >= args.max_new_tokens:
            print(f"⚠ Hit max_new_tokens limit ({args.max_new_tokens}), did NOT hit EOS token")
            print(f"  This means the model did NOT generate any stop token (128009 or 128001)")
        else:
            print(f"⚠ Unexpected stop condition")
    
    # Clean up special tokens for display
    generated_text = _truncate_generated_text(generated_text)
    
    # Also show decoded text with special tokens visible for debugging
    print(f"\n=== Generated (with special tokens visible) ===")
    print(repr(generated_text_with_special))

    print("=== Prompt ===")
    print(json.dumps(messages, ensure_ascii=False, indent=2))
    print("\n=== Generated ===")
    print(generated_text)


if __name__ == "__main__":
    main()