#!/bin/bash
#SBATCH -J megatron
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --output=gpt_1n2g_test.log

# Vars with defaults
NEXP="${NEXP:-1}"


module unload cray-dsmml cray-libsci perftools-base
module load nccl/2.18.3-cu12 cudnn/8.9.3_cuda12

export PREFIX=/mscratch/sd/s/schheda/summer24/torch
export LD_LIBRARY_PATH=${PREFIX}/opt2/lib:${LD_LIBRARY_PATH}
export PATH=${PREFIX}/opt2/bin:${PATH}:/sbin
source ${PREFIX}/env2/bin/activate

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd
# export OMP_NUM_THREADS=1

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export GROUP_RANK=$SLURM_NODEID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export MASTER_PORT=29500

# Extra command line args
# args=$@

# check where device assertions were triggered
export CUDA_LAUNCH_BLOCKING=1

# Run experiments
for iexp in $(seq 1 "${NEXP}"); do

    echo "Beginning trial ${iexp} of ${NEXP}"

    # Run experiment
    # export SEED=${_seed_override:-$RANDOM}
    # srun -u --mpi=pmi2 shifter \
    srun -u --mpi=pmi2 \
        python3 -u run_simple_mcore_train_loop.py

done

