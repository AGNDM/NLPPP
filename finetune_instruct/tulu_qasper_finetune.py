import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

def main():
    # 1. Load Dataset
    dataset_path = "./data/qasper/processed/finetuning_dataset_final.parquet"
    print(f"Loading dataset from {dataset_path}...")
    # Assuming we're using the data as the train split
    dataset = load_dataset("parquet", data_files={"train": dataset_path}, split="train")

    # 2. Load Model and Tokenizer
    model_name = "allenai/Llama-3.1-Tulu-3-8B"

    print(f"Loading tokenizer {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token. Setting pad_token to eos_token for Llama 3 generation compatibility.")
        tokenizer.pad_token = tokenizer.eos_token

    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    print(f"Loading model {model_name} on device {local_rank}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": local_rank}, # Use specific GPU for DistributedDataParallel (DDP)
        torch_dtype=torch.bfloat16, # Recommended for Llama 3 models if GPU supports it
    )

    # Convert the dataset to the model's chat format
    print("Formatting dataset with chat template...")
    def create_prompt(example):
        messages = [
            {"role": "user", "content": str(example["input"])},
            {"role": "assistant", "content": example["answer"]}
        ]
        # Use Tulu 3's built-in chat template to format the conversation properly
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(create_prompt, remove_columns=dataset.column_names)

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 4. Training Arguments
    training_args = SFTConfig(
        output_dir="./tulu_qasper_lora_output",
        per_device_train_batch_size=8, # GH200 has 96GB/144GB VRAM, can use a larger batch size
        gradient_accumulation_steps=1, # Adjust if you want larger effective batch size
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=1, # Adjust as per your dataset size
        save_strategy="epoch",
        bf16=True, # GH200 supports bfloat16 perfectly
        report_to="none", # Switch to "wandb" or "tensorboard" if you use them
        remove_unused_columns=False,
        ddp_find_unused_parameters=False, # Required for LoRA training with DDP
        gradient_checkpointing=True, # Save memory for large models
        dataset_text_field="text",
        max_seq_length=2048,
    )

    # 5. Initialize SFTTrainer
    print("Setting up DataCollatorForCompletionOnlyLM...")
    # Tulu 3 / Llama 3 chat format uses a specific tag before the assistant's reply.
    response_template_str = "<|assistant|>\n"
    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=training_args,
        data_collator=collator, # Apply completion-only loss
    )

    # 6. Start Training
    print("Starting training...")
    print(collator([dataset[0]])["labels"][0])
    trainer.train()

    # 7. Save the final model
    print("Saving final LoRA adapters...")
    trainer.save_model("./tulu_qasper_lora_final")

if __name__ == "__main__":
    main()
