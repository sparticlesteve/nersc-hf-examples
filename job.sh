#!/bin/bash
#SBATCH -J hf-accelerate-example
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time 10

# Environment setup
#module use /global/homes/s/sfarrell/WorkAreas/software/modulefiles/src
#module load pytorch/2.1.0-cu12
#module load pytorch/2.1.0-cu12 gcc-native/12.3
#module load pytorch/2.0.1
module load pytorch/2.3.1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29507
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

set -x

# Set the HF cache directory
export HF_HOME=$SCRATCH/cache/huggingface

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
