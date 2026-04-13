from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from transformers import clone_chat_template
except ImportError:
    from transformers.utils import clone_chat_template
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import torch
import json
import os
import warnings
warnings.filterwarnings('ignore')

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
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
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    model, tokenizer, added_tokens = clone_chat_template(
        model,
        tokenizer,
        source_tokenizer_path="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA adapter
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=['up_proj','down_proj','gate_proj','k_proj','q_proj','v_proj','o_proj']
    )
    model = get_peft_model(model, peft_config)
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

    ds = ds.map(fmt, num_proc=4)
    ds = ds.train_test_split(test_size=0.01, seed=42)

    # SFT Trainer
    sft_config = SFTConfig(
        output_dir=run_output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        disable_tqdm=False,
        optim="paged_adamw_32bit",
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        warmup_steps=50,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False, bf16=True,
        gradient_checkpointing=True,
        group_by_length=True,
        dataloader_num_workers=4,
        dataset_text_field="text",
        max_length=512,
        packing=False,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=10,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        peft_config=peft_config,
        args=sft_config,
    )

    print("Starting Training :")
    trainer.train()
    # 1) Save the LoRA adapter alone (optional)
    adapter_dir = os.path.join(output_root, new_model)
    trainer.model.save_pretrained(adapter_dir)
    print(f"Adapter-only weights saved to '{adapter_dir}'")

    # 2) Merge LoRA adapters into the base model & save full checkpoint
    print("Merging adapters into base weights...")
    # trainer.model is a PeftModel; merge_and_unload() returns a pure transformers model
    merged_model = trainer.model.merge_and_unload()
    full_model_dir = os.path.join(output_root, f"{new_model}-full")
    merged_model.save_pretrained(full_model_dir)
    tokenizer.save_pretrained(full_model_dir)
    print(f"Full merged model saved to '{full_model_dir}'")
    
if __name__ == "__main__":
    
    main()