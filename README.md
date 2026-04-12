# NLPPP

## Fine-tuned Model

- Fine-tuned Qwen-0.5B available on Huggingface: https://huggingface.co/AGNDM/Fine-tuned_NLP_Qwen_0.5B
- More models coming soon

## Fine-tune Llama 8B on Tulu

Run with Accelerate (single node, multi-GPU):

```bash
accelerate launch --num_processes=4 --num_machines=1 --machine_rank=0 \
	finetune/finetuning_tulu.py \
	--model_name meta-llama/Meta-Llama-3-8B-Instruct \
	--dataset_name allenai/tulu-v2-sft-mixture \
	--train_split train \
	--output_dir finetune/output_llama_tulu_8B \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--num_train_epochs 2 \
	--learning_rate 2e-5 \
	--max_seq_length 2048
```

For a local dataset file (`.json`, `.jsonl`, `.parquet`), replace `--dataset_name ...` with:

```bash
--data_path /path/to/tulu_dataset.parquet
```
