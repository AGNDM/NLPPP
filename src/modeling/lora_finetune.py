import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

device = "cpu"
def load_qa_data(pairs_file: str):
    """Extract question and answer from (question, chunk, answer) pairs"""
    with open(pairs_file, "r") as f:
        pairs = json.load(f)
    # Deduplication (optional)
    seen = set()
    unique_pairs = []
    for p in pairs:
        key = (p["question"], p["answer"])
        if key not in seen:
            seen.add(key)
            unique_pairs.append({"question": p["question"], "answer": p["answer"]})
    print(f"Loaded {len(unique_pairs)} QA pairs (after deduplication)")
    return unique_pairs

def preprocess_function(examples, tokenizer, max_input_length=512, max_target_length=128):
    """Convert questions and answers to model input format"""
    inputs = examples["question"]
    targets = examples["answer"]
    
    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True, 
        padding="max_length"
    )
    labels = tokenizer(
        targets, 
        max_length=max_target_length, 
        truncation=True, 
        padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def train_lora_qa(
    model_name="google/flan-t5-small",  # Use small for faster training, base also works
    train_file="data/processed/train_pairs.json",
    val_file="data/processed/val_pairs.json",
    output_dir="./models/lora_qa_final"
):
    # 1. Load data
    print("Loading training data...")
    train_data = load_qa_data(train_file)
    val_data = load_qa_data(val_file)
    
    # 2. Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    # 3. Load base model and tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    
    # 4. Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q", "v"]  # T5 model uses "q", "v"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 5. Preprocess datasets
    print("Preprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # 6. Set training arguments
    training_args = TrainingArguments(
        output_dir="./models/lora_qa_checkpoints",
        per_device_train_batch_size=4,  # Reduce batch size to avoid insufficient memory
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=50,
        learning_rate=3e-4,
        fp16=False,  # Intel Mac does not support fp16
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        no_cuda=True,
        dataloader_pin_memory=False
    )
    
    # 7. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    trainer.train()
    
    # 8. Save model
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training completed!")

if __name__ == "__main__":
    # Run fine-tuning
    train_lora_qa()