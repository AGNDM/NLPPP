#!/usr/bin/env python3
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)

print("=" * 60)
print("TOKEN DIAGNOSTIC")
print("=" * 60)
print(f"eos_token: {repr(tokenizer.eos_token)}")
print(f"eos_token_id: {tokenizer.eos_token_id}")
print(f"pad_token: {repr(tokenizer.pad_token)}")
print(f"pad_token_id: {tokenizer.pad_token_id}")

eot_ids = tokenizer.convert_tokens_to_ids(["<|eot_id|>"])
print(f"\nconvert_tokens_to_ids(['<|eot_id|>']): {eot_ids}")

encoded_eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
print(f"encode('<|eot_id|>', add_special_tokens=False): {encoded_eot}")

if encoded_eot:
    decoded = tokenizer.decode(encoded_eot)
    print(f"decode({encoded_eot}): {repr(decoded)}")

begin_ids = tokenizer.encode("<|begin_of_text|>", add_special_tokens=False)
print(f"\nencode('<|begin_of_text|>'): {begin_ids}")

# Check if this is the issue - maybe these special tokens aren't in vocab
print(f"\nVocab size: {len(tokenizer)}")
print(f"unk_token_id: {tokenizer.unk_token_id}")
