#!/bin/bash
#SBATCH -J hf-accelerate-example
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time 10

# Environment setup
module use /global/common/software/nersc/pe/modulefiles/latest
module load pytorch/2.6.0

# Use scratch for huggingface cache
export HF_HOME=$SCRATCH/cache/huggingface

# Distributed training configuration
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507

# Other settings
export OMP_NUM_THREADS=8
#export NCCL_DEBUG=INFO

set -x

# Using HF Accelerate Launch
srun -l bash -c "
    accelerate launch \
    --num_machines $SLURM_JOB_NUM_NODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --num_processes $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE)) \
    --multi_gpu \
    --debug \
    --rdzv_backend c10d \
    nlp_example.py
"

# Using torchrun
#srun -u torchrun \
#    --nnodes=$SLURM_JOB_NUM_NODES \
#    --nproc-per-node=$SLURM_GPUS_PER_NODE \
#    --rdzv-backend=c10d \
#    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#    nlp_example.py

echo "SUCCESS"
