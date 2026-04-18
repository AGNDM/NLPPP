#!/bin/bash
#SBATCH --job-name=tulu_qasper_finetune
#SBATCH --account=a131
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --output=tulu_qasper_%j.out # 标准输出日志 (%j 为 Job ID)
#SBATCH --error=tulu_qasper_%j.err # 错误输出日志

# 加载你的虚拟环境
source $SCRATCH/share/xianrong_liu/.venv/bin/activate
source $SCRATCH/share/xianrong_liu/cache_env_setup.sh

# 切换到工作目录
cd $SCRATCH/share/xianrong_liu/tutorial_NLPPP

# 提升多线程性能
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting distributed training on 4 GPUs..."

# 使用 torchrun 启动 4 卡训练
srun --environment=$SCRATCH/share/xianrong_liu/tut/ngc-pytorch-25.06.toml bash -c "
    source $SCRATCH/share/xianrong_liu/myvenv/bin/activate && \
    source $SCRATCH/share/xianrong_liu/cache_env_setup.sh && \
    pip install datasets && \
    cd tutorial_NLPPP && \
    torchrun --nproc_per_node=4 finetune_instruct/tulu_qasper_finetune.py
"
