#!/usr/bin/env bash
# Launch Qwen-0.5B finetuning on 4 GH200 GPUs using Accelerate
# Ensure you have activated the conda environment `py313` before running.

# Activate conda environment (uncomment if needed)
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate py313

# Install accelerate config if not done
# accelerate config

# Run training
accelerate launch \
    --num_processes 4 \
    --gpu_ids 0 1 2 3 \
    --mixed_precision bf16 \
    ./finetune/train_qwen_0.5B_qasper.py \
    --data_path data/qasper/processed/tinetuning_dataset.parquet \
    --output_dir finetune/output_qwen_0.5B \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --max_seq_length 1024 \
    --samples_output finetune/samples.txt \
    --num_samples 5
