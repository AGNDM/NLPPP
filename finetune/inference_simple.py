import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Load Llama-8B base + LoRA adapter and run one prompt inference."
	)
	parser.add_argument(
		"--model_name",
		type=str,
		default="meta-llama/Meta-Llama-3-8B",
		help="Base model name or path.",
	)
	parser.add_argument(
		"--adapter_path",
		type=str,
		required=True,
		help="Path to LoRA adapter directory (must contain adapter_config.json).",
	)
	parser.add_argument(
		"--prompt",
		type=str,
		required=True,
		help="Single user prompt.",
	)
	parser.add_argument("--max_new_tokens", type=int, default=256)
	parser.add_argument("--temperature", type=float, default=0.7)
	parser.add_argument("--top_p", type=float, default=0.9)
	return parser.parse_args()


def build_single_turn_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> tuple[str, str]:
	messages = [{"role": "user", "content": user_prompt}]
	assistant_prefix = "assistant\n"

	if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
		prompt_text = tokenizer.apply_chat_template(
			messages,
			tokenize=False,
			add_generation_prompt=True,
		)
		return prompt_text, assistant_prefix

	prompt_text = (
		"<|start_header_id|>user<|end_header_id|>\n\n"
		f"{user_prompt}<|eot_id|>"
		"<|start_header_id|>assistant<|end_header_id|>\n\n"
	)
	assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
	return prompt_text, assistant_prefix


def load_model_and_tokenizer(model_name: str, adapter_path: str):
	adapter_config = Path(adapter_path) / "adapter_config.json"
	if not adapter_config.exists():
		raise FileNotFoundError(
			f"{adapter_config} not found. This script only supports LoRA adapters."
		)

	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
	if tokenizer.pad_token is None and tokenizer.eos_token is not None:
		tokenizer.pad_token = tokenizer.eos_token

	dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
	base_model = AutoModelForCausalLM.from_pretrained(
		model_name,
		torch_dtype=dtype,
		device_map="auto",
	)
	model = PeftModel.from_pretrained(base_model, adapter_path)
	model.eval()
	return tokenizer, model


def main() -> None:
	args = parse_args()
	prompt = args.prompt.strip()
	if not prompt:
		raise ValueError("--prompt cannot be empty.")

	tokenizer, model = load_model_and_tokenizer(args.model_name, args.adapter_path)
	prompt_text, assistant_prefix = build_single_turn_prompt(tokenizer, prompt)

	inputs = tokenizer(prompt_text, return_tensors="pt")
	device = model.device if hasattr(model, "device") else next(model.parameters()).device
	inputs = {key: value.to(device) for key, value in inputs.items()}

	do_sample = args.temperature > 0
	generation_kwargs = {
		"max_new_tokens": args.max_new_tokens,
		"do_sample": do_sample,
		"temperature": args.temperature if do_sample else None,
		"top_p": args.top_p if do_sample else None,
		"pad_token_id": tokenizer.pad_token_id,
		"eos_token_id": tokenizer.eos_token_id,
	}
	generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}

	with torch.no_grad():
		generated_ids = model.generate(**inputs, **generation_kwargs)

	prompt_len = inputs["input_ids"].shape[-1]
	new_token_ids = generated_ids[0][prompt_len:]
	output_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

	print(f"{assistant_prefix}{output_text}")


if __name__ == "__main__":
	main()
