#!/bin/bash
#SBATCH -C gpu
#SBATCH -t 30
#SBATCH -q debug

. setup.sh

OUTPUT_DIR=$SCRATCH/hf-deepspeed-example
export OMP_NUM_THREADS=1

accelerate launch --config_file=accelerate_config.yaml \
    deepspeed_with_config_support.py \
    --config_name "gpt2-large" \
    --tokenizer_name "gpt2-large" \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --block_size 128 \
    --output_dir $OUTPUT_DIR \
    --learning_rate 5e-4 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --num_train_epochs 1
