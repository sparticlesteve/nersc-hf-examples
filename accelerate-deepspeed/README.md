This example shows how to run a huggingface accelerate + deepspeed zero example
on Perlmutter. It is currently only single node but can be extended easily.

## Code
This example was taken from https://huggingface.co/docs/accelerate/usage_guides/deepspeed#deepspeed-config-file
on 2025-03-06.

The script is from here: https://github.com/huggingface/accelerate/blob/main/examples/by_feature/fsdp_with_peak_mem_tracking.py

The deepspeed config is from here: https://github.com/huggingface/accelerate/blob/main/examples/deepspeed_config_templates/zero_stage3_config.json

## First time setup

```bash
. setup.sh
pip install deepspeed
```

## Submit job

```bash
sbatch job.sh
```
