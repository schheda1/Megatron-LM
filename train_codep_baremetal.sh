#!/bin/bash
#SBATCH -J megatron
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --output=code_parrot.log

module unload cray-dsmml cray-libsci perftools-base gpu/1.0
module load cudatoolkit/12.2 nccl/2.21.5 cudnn/9.1.0

export PREFIX=${SCRATCH}/summer24/torch
export LD_LIBRARY_PATH=${PREFIX}/opt/lib:${LD_LIBRARY_PATH}
export PATH=${PREFIX}/opt/bin:${PATH}:/sbin
source ${PREFIX}/env/bin/activate

# Other settings
export MASTER_ADDR=$SLURM_LAUNCH_NODE_IPADDR
export FI_MR_CACHE_MONITOR=userfaultfd


export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
# export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export MASTER_PORT=29500
export MPICH_GPU_SUPPORT_ENABLED=0

# export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

# data stuff + checkpoint dir
export CODE_PARROT_ROOT=${SCRATCH}/summer24/code_parrot
export VOCAB_FILE=${CODE_PARROT_ROOT}/gpt2-vocab.json
export MERGE_FILE=${CODE_PARROT_ROOT}/gpt2-merges.txt
export CHECKPOINT_PATH=${CODE_PARROT_ROOT}/ckpt
export DATA_PATH=${CODE_PARROT_ROOT}/codeparrot_content_document

# GPT model args
GPT_ARGS=(
--num-layers 96
--hidden-size 12288
--num-attention-heads 96
--seq-length 2048
--max-position-embeddings 2048
--micro-batch-size 1
--global-batch-size 768
--lr 0.0005
--train-iters 20
--lr-decay-iters 150
--weight-decay 0.1
--log-interval 5
--save-interval 20 # save-interval = train-iters, get progress.txt to dump throughput, etc.
--save ${CHECKPOINT_PATH}
--load ${CHECKPOINT_PATH}
--eval-iters 10
) 

# srun -u --mpi=pmi2 shifter \
# export PYTHONPATH=${SCRATCH}/summer24/sc-Megatron-LM:${PYTHONPATH} 
srun -u  \
     python3 -u  pretrain_gpt.py \
     	--tensor-model-parallel-size 8 \
	--pipeline-model-parallel-size 8 \
	--overlap-grad-reduce \
	--overlap-param-gather \
	--use-distributed-optimizer \
	--fp16 \
	--use-flash-attn \
	--distributed-backend nccl \
	${GPT_ARGS[@]} \
	--vocab-file ${VOCAB_FILE} \
	--merge-file ${MERGE_FILE} \
	--data-path ${DATA_PATH} \
	--num-dataset-builder-threads 24 \
	--log-throughput \
	--log-progress \
	--split 969,30,1

# clean up checkpoints dir. Since checkpoints are stored on parallel fs, 1 task on 1 node will do
# srun --nnodes 1 -n 1 rm -rf ${CHECKPOINT_PATH}/*

