# Source me

# Load latest pytorch module
module use /global/common/software/nersc/pe/modulefiles/latest
module load pytorch/2.6.0

# Use scratch for huggingface and triton caches
export HF_HOME=$SCRATCH/cache/huggingface
export TRITON_CACHE_DIR=$SCRATCH/cache/triton

# For compilation of extensions
export CXX=g++
export CC=gcc

# If first time, source this script and then do
# pip install deepspeed
