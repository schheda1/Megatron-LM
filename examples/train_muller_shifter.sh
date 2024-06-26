#!/bin/bash
#SBATCH -J megatron
#SBATCH -A nintern
#SBATCH -C gpu
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=schheda/megatron:latest
#SBATCH --module=gpu
#SBATCH --output=latest.log

# Vars with defaults
NEXP="${NEXP:-1}"

# module load nccl/2.19.4
# export MPICH_GPU_SUPPORT_ENABLED=0

# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_DEBUG=INFO
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Other settings
export MASTER_ADDR=$(hostname)
export FI_MR_CACHE_MONITOR=userfaultfd


# Extra command line args
args=$@


# Run experiments
for iexp in $(seq 1 "${NEXP}"); do

    echo "Beginning trial ${iexp} of ${NEXP}"

    # Run experiment
    #export SEED=${_seed_override:-$RANDOM}
    srun -u  shifter \
	bash run_mcore.sh

done

