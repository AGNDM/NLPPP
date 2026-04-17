#!/bin/bash
#SBATCH --job-name=tulu-ft-128r
#SBATCH --account=a131
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

: "${NGC_ENV_FILE:=$SCRATCH/share/xianrong_liu/tut/ngc-pytorch-25.06.toml}"
: "${PROJECT_DIR:=$./tutorial_NLPPP}"
: "${VENV_ACTIVATE:=$./myvenv/bin/activate}"
: "${CACHE_ENV_SETUP:=$./cache_env_setup.sh}"

srun -A a131 --environment="$NGC_ENV_FILE" bash -lc "
  source '$VENV_ACTIVATE' && \
  source '$CACHE_ENV_SETUP' && \
  cd '$PROJECT_DIR' && \
  accelerate launch --multi_gpu --num_processes 4 --num_machines 1 --machine_rank 0 --mixed_precision bf16 finetune_instruct/tulu_qasper_finetune.py
"
