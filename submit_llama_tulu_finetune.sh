#!/bin/bash
#SBATCH --job-name=llama8b-tulu-ft
#SBATCH --account=a131
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

# Run from this script's directory (repo root)
cd /iopsstor/scratch/cscs/tong/share/xianrong_liu/tutorial_NLPPP

# Optional: override these with exported env vars before sbatch
: "${UENV_IMAGE:=pytorch/v2.8.0:v1}"
: "${UENV_VIEW:=default}"
: "${VENV_PATH:=/iopsstor/scratch/cscs/tong/share/xianrong_liu/.venv/bin/activate}"
: "${EXTRA_ENV:=/iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt/cache_env_setup.sh}"

# source /iopsstor/scratch/cscs/tong/share/xianrong_liu/cache_env_setup.sh

HF_TOKEN="hf_xxx"  # Replace with your actual token or export HF_TOKEN before sbatch
# Required for gated model download (meta-llama/*)
: "${HF_TOKEN:?Please export HF_TOKEN before sbatch, e.g. export HF_TOKEN=hf_xxx}"
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

# Launch 4-process distributed training with Accelerate
srun --uenv="${UENV_IMAGE}" --view="${UENV_VIEW}" bash -c "\
    if [ -f \"${VENV_PATH}\" ]; then source \"${VENV_PATH}\"; fi && \
    if [ -f \"${EXTRA_ENV}\" ]; then source \"${EXTRA_ENV}\"; fi && \
    export HF_TOKEN=\"${HF_TOKEN}\" && \
    export HUGGINGFACE_HUB_TOKEN=\"${HF_TOKEN}\" && \
    accelerate launch --num_processes=4 --num_machines=1 --machine_rank=0 \
      ./finetune/finetuning_tulu.py \
      --model_name meta-llama/Meta-Llama-3-8B-Instruct \
      --dataset_name allenai/tulu-3-sft-mixture \
      --train_split train \
      --output_dir finetune/output_llama_tulu_8B \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 16 \
      --num_train_epochs 2 \
      --learning_rate 2e-5 \
      --max_seq_length 2048\
"
