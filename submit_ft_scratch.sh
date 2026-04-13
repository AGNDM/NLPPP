#!/bin/bash
#SBATCH --job-name=llama8b-ft-scratch
#SBATCH --account=a131
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=log/%x-%j.out
#SBATCH --error=log/%x-%j.err

set -euo pipefail

: "${BASE_DIR:=$SCRATCH/share/xianrong_liu}"
: "${NGC_ENV_FILE:=$SCRATCH/share/xianrong_liu/tut/ngc-pytorch-25.06.toml}"
: "${PROJECT_DIR:=$BASE_DIR/tutorial_NLPPP}"
: "${VENV_ACTIVATE:=$BASE_DIR/myvenv/bin/activate}"
: "${CACHE_ENV_SETUP:=$BASE_DIR/cache_env_setup.sh}"

# Optional training knobs for ft_scratch.py (can be overridden at submit time)
: "${PER_DEVICE_TRAIN_BATCH_SIZE:=4}"
: "${PER_DEVICE_EVAL_BATCH_SIZE:=4}"
: "${GRADIENT_ACCUMULATION_STEPS:=1}"
: "${MAX_SEQ_LEN:=1024}"
: "${USE_PACKING:=false}"
: "${USE_GRADIENT_CHECKPOINTING:=true}"

mkdir -p "$PROJECT_DIR/log"

srun -A a131 --environment="$NGC_ENV_FILE" bash -lc "
  source '$VENV_ACTIVATE' && \
  source '$CACHE_ENV_SETUP' && \
  cd '$PROJECT_DIR' && \
  export PER_DEVICE_TRAIN_BATCH_SIZE='$PER_DEVICE_TRAIN_BATCH_SIZE' && \
  export PER_DEVICE_EVAL_BATCH_SIZE='$PER_DEVICE_EVAL_BATCH_SIZE' && \
  export GRADIENT_ACCUMULATION_STEPS='$GRADIENT_ACCUMULATION_STEPS' && \
  export MAX_SEQ_LEN='$MAX_SEQ_LEN' && \
  export USE_PACKING='$USE_PACKING' && \
  export USE_GRADIENT_CHECKPOINTING='$USE_GRADIENT_CHECKPOINTING' && \
  accelerate launch --num_processes=4 --num_machines=1 --machine_rank=0 finetune/ft_scratch.py
"
