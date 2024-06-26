source /global/common/software/nersc9/nccl/2.19.4/env_nccl.sh

# NCCL 2.21.5 local
# source /mscratch/sd/s/schheda/summer24/nccl/env_nccl.sh

# export MPICH_GPU_SUPPORT_ENABLED=0

PYTHONPATH=/workspace/Megatron-LM:${PYTHONPATH} python3 -u run_simple_mcore_train_loop.py
