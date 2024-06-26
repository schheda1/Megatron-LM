#!/bin/bash
#SBATCH -J megatron
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --output=gpt_1n2g_test.log

# Vars with defaults
NEXP="${NEXP:-1}"


module unload cray-dsmml cray-libsci perftools-base
module load nccl/2.21.5 cudnn/9.1.0

export PREFIX=/mscratch/sd/s/schheda/summer24/torch
export LD_LIBRARY_PATH=${PREFIX}/opt/lib:${LD_LIBRARY_PATH}
export PATH=${PREFIX}/opt/bin:${PATH}:/sbin
source ${PREFIX}/env/bin/activate

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd
# export OMP_NUM_THREADS=1
# 12 dataloader workers work best

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export MASTER_PORT=29500

export MPICH_GPU_SUPPORT_ENABLED=0

# Extra command line args
# args=$@

# check where device assertions were triggered
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=INFO

# Run experiments
for iexp in $(seq 1 "${NEXP}"); do

    echo "Beginning trial ${iexp} of ${NEXP}"

    # Run experiment
    # export SEED=${_seed_override:-$RANDOM}
    # srun -u --mpi=pmi2 shifter \
    PYTHONPATH=${SCRATCH}/summer24/Megatron-LM:${PYTHONPATH} srun -u  \
	    python3 -u run_simple_mcore_train_loop.py

done

