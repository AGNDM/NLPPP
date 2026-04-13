from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig, clone_chat_template, DataCollatorForCompletionOnlyLM
import torch
import json
import os
import inspect
import warnings
warnings.filterwarnings('ignore')

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print("Setting UP Configs")
    # QLoRA config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    new_model = "llama-3-8b-NLPPP-general"
    output_root = "outputs"
    run_output_dir = os.path.join(output_root, new_model)
    os.makedirs(output_root, exist_ok=True)
    # Load & prepare model
    requested_attn_impl = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")
    flash_attn_enabled = False
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=requested_attn_impl,
        )
        flash_attn_enabled = requested_attn_impl in {
            "flash_attention_2",
            "flash_attention_3",
            "kernels-community/flash-attn2",
            "kernels-community/flash-attn3",
            "kernels-community/vllm-flash-attn3",
        }
        print(f"Using attention implementation: {requested_attn_impl}")
    except Exception as exc:
        print(f"Failed to load with attn_implementation={requested_attn_impl}: {exc}")
        print("Falling back to default attention implementation.")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    source_tokenizer_path = "meta-llama/Meta-Llama-3-8B-Instruct"
    model, tokenizer, added_tokens = clone_chat_template(
        model,
        tokenizer,
        source_tokenizer_path=source_tokenizer_path,
    )

    #if not getattr(tokenizer, "chat_template", None):
    #    instruct_tokenizer = AutoTokenizer.from_pretrained(source_tokenizer_path)
    #    tokenizer.chat_template = getattr(instruct_tokenizer, "chat_template", None)

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("chat_template is still missing after cloning/copying from instruct tokenizer")

    if isinstance(added_tokens, int):
        added_token_count = added_tokens
    elif added_tokens is None:
        added_token_count = 0
    else:
        added_token_count = len(added_tokens)

    print(f"chat_template ready: {bool(tokenizer.chat_template)}")
    print(f"tokens added by clone_chat_template: {added_token_count}")
    if added_token_count > 0:
        print("WARNING: New tokens were added. With LoRA-only training, new token embeddings are not fully trained.")

    if tokenizer.pad_token is None:
        llama3_right_pad_token = "<|finetune_right_pad_id|>"
        if llama3_right_pad_token in tokenizer.get_vocab():
            tokenizer.pad_token = llama3_right_pad_token
            print(f"Using existing tokenizer pad token: {tokenizer.pad_token}")
        else:
            added_pad = tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            if added_pad > 0:
                model.resize_token_embeddings(len(tokenizer))
                print("Added new pad token '<|pad|>' and resized model embeddings.")
            print(f"Using tokenizer pad token: {tokenizer.pad_token}")
    model.config.pad_token_id = tokenizer.pad_token_id

    # Throughput knobs (override via environment variables if needed)
    per_device_train_batch_size = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE", "4"))
    per_device_eval_batch_size = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE", str(per_device_train_batch_size)))
    gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "1"))
    max_seq_len = int(os.getenv("MAX_SEQ_LEN", "1024"))
    use_packing = False # Forced to False when using DataCollatorForCompletionOnlyLM
    use_gradient_checkpointing = os.getenv("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true"

    if use_packing and not flash_attn_enabled:
        print("WARNING: packing requested but flash attention is unavailable; disabling packing for safety.")
        use_packing = False

    print(
        f"Train config: bs={per_device_train_batch_size}, eval_bs={per_device_eval_batch_size}, "
        f"ga={gradient_accumulation_steps}, max_seq_len={max_seq_len}, "
        f"packing={use_packing}, grad_ckpt={use_gradient_checkpointing}, "
        f"attn_impl={'flash' if flash_attn_enabled else 'default'}"
    )

    # LoRA adapter
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=['up_proj','down_proj','gate_proj','k_proj','q_proj','v_proj','o_proj'],
        modules_to_save=["embed_tokens", "lm_head"]
    )
    print("Loading and setting up Dataset for Training")
    # Dataset prep
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="all")
    # Only use a fraction of the data
    sample_size = max(1, int(len(ds) * 0.1))
    ds = ds.shuffle(seed=42).select(range(sample_size))

    def fmt(r):
        msgs = r.get("messages", [])

        # Some exports may store messages as a JSON string instead of a Python list
        if isinstance(msgs, str):
            msgs = json.loads(msgs)

        # Keep only turns with both role/content, matching chat template expectations
        msgs = [m for m in msgs if isinstance(m, dict) and "role" in m and "content" in m]

        r["text"] = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
        return r

    original_columns = ds.column_names
    ds = ds.map(fmt, num_proc=4, remove_columns=original_columns)
    ds = ds.train_test_split(test_size=0.01, seed=42)

    # SFT Trainer (compatible across TRL/Transformers versions)
    sft_kwargs = {
        "output_dir": run_output_dir,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "disable_tqdm": False,
        "optim": "paged_adamw_32bit",
        "num_train_epochs": 3,
        "eval_steps": 100,
        "logging_steps": 10,
        "warmup_steps": 50,
        "logging_strategy": "steps",
        "learning_rate": 2e-4,
        "fp16": False,
        "bf16": True,
        "gradient_checkpointing": use_gradient_checkpointing,
        "group_by_length": True,
        "dataloader_num_workers": 16,
        "dataset_text_field": "text",
        "packing": use_packing,
        "save_strategy": "steps",
        "save_steps": 1000,
        "save_total_limit": 10,
    }

    sft_init_params = inspect.signature(SFTConfig.__init__).parameters
    strategy_value = "steps"
    if "evaluation_strategy" in sft_init_params:
        sft_kwargs["evaluation_strategy"] = strategy_value
    elif "eval_strategy" in sft_init_params:
        sft_kwargs["eval_strategy"] = strategy_value

    if "max_length" in sft_init_params:
        sft_kwargs["max_length"] = max_seq_len
    elif "max_seq_length" in sft_init_params:
        sft_kwargs["max_seq_length"] = max_seq_len

    sft_kwargs = {key: value for key, value in sft_kwargs.items() if key in sft_init_params}
    sft_config = SFTConfig(**sft_kwargs)

    # Use DataCollatorForCompletionOnlyLM to compute loss only on assistant messages
    response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "train_dataset": ds["train"],
        "eval_dataset": ds["test"],
        "peft_config": peft_config,
        "args": sft_config,
        "data_collator": collator,
    }
    sft_trainer_init_params = inspect.signature(SFTTrainer.__init__).parameters
    if "tokenizer" in sft_trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    elif "processing_class" in sft_trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting Training :")
    trainer.train()
    # 1) Save the LoRA adapter alone
    adapter_dir = os.path.join(output_root, new_model)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter-only weights saved to '{adapter_dir}'")

    # Clear VRAM before full-precision merge
    del model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2) Reload full-precision base model, attach adapter, merge, and save full checkpoint
    print("Merging adapters into base weights...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.resize_token_embeddings(len(tokenizer))

    model_to_merge = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = model_to_merge.merge_and_unload()
    full_model_dir = os.path.join(output_root, f"{new_model}-full")
    merged_model.save_pretrained(full_model_dir)
    tokenizer.save_pretrained(full_model_dir)
    print(f"Full merged model saved to '{full_model_dir}'")
    
if __name__ == "__main__":
    
    main()