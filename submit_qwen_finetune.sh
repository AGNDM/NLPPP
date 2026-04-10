#!/bin/bash
#SBATCH --job-name=qwen05b-qasper-ft
#SBATCH --account=a131
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

# Run from this script's directory (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p log

# Optional: override these with exported env vars before sbatch
: "${UENV_IMAGE:=pytorch/v2.8.0:v1}"
: "${UENV_VIEW:=default}"
: "${VENV_PATH:=/iopsstor/scratch/cscs/tong/share/xianrong_liu/.venv/bin/activate}"
: "${EXTRA_ENV:=/iopsstor/scratch/cscs/tong/share/xianrong_liu/build-nanogpt/cache_env_setup.sh}"

# Launch 4-process distributed training with Accelerate
srun --uenv="${UENV_IMAGE}" --view="${UENV_VIEW}" bash -c "\
    if [ -f \"${VENV_PATH}\" ]; then source \"${VENV_PATH}\"; fi && \
    if [ -f \"${EXTRA_ENV}\" ]; then source \"${EXTRA_ENV}\"; fi && \
    accelerate launch --num_processes=4 --num_machines=1 --machine_rank=0 \
      finetune/train_qwen_0.5B_qasper.py \
      --data_path data/qasper/processed/finetuning_dataset.parquet \
      --model_name Qwen/Qwen2.5-0.5B-Instruct \
      --output_dir finetune/output_qwen_0.5B \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 16 \
      --num_train_epochs 3 \
      --learning_rate 2e-5 \
      --max_seq_length 1024 \
      --num_samples 5 \
      --samples_output finetune/samples_qwen_0.5B.txt\
"
