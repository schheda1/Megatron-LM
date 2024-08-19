#!/bin/bash
#SBATCH -J mcore-r9
#SBATCH -A nintern
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH -N 128
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --image=schheda/megatron:latest
#SBATCH --module=gpu
#SBATCH --output=08-02-2024/r9.log

# parallel args
export TP_SIZE=4
export PP_SIZE=8
export CP_SIZE=2

export MAPPING="cp-tp-pp-dp"  # default mapping
export GLOBAL_BATCH_SIZE=1024
export MICRO_BATCH_SIZE=1
export TIMER_LEVEL=0

#Model args: ViT style
export NUM_LAYERS=48
export HIDDEN_SIZE=6144
export NUM_ATTN_HEADS=32
export SEQUENCE_LEN=32400

# Other settings
export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR
export FI_MR_CACHE_MONITOR=userfaultfd
export WORLD_SIZE=$SLURM_NTASKS

# export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# data stuff + checkpoint dir
export CODE_PARROT_ROOT=${SCRATCH}/summer24/code_parrot
export VOCAB_FILE=${CODE_PARROT_ROOT}/gpt2-vocab.json
export MERGE_FILE=${CODE_PARROT_ROOT}/gpt2-merges.txt
export CHECKPOINT_PATH=${CODE_PARROT_ROOT}/ckpt
export DATA_PATH=${CODE_PARROT_ROOT}/codeparrot_content_document

# cleanup cache and any remnants in checkpoint dir
find . -type d -name "__pycache__" -exec rm -rf {} +
srun --nodes 1 -n 1 rm -rf ${CHECKPOINT_PATH}/*

# parallel args contd.
export DP_SIZE=$(( WORLD_SIZE / (TP_SIZE*PP_SIZE)))

# srun -u --mpi=pmi2 --kill-on-bad-exit=0 shifter \
srun -u --mpi=pmi2 \
	shifter --module=gpu \
		bash run_codep.sh

# clean up checkpoints dir. Since checkpoints are stored on parallel fs, 1 task on 1 node will do
# srun --nnodes 1 -n 1 rm -rf ${CHECKPOINT_PATH}/*

