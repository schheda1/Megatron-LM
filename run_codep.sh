# Perlmutter + Shifter specific NCCL plugin.
# Comment it out if not on PM or modify if a newer env script is available 
source /global/common/software/nersc9/nccl/2.19.4/env_nccl.sh

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real
export TRITON_CACHE_DIR="/dev/shm/rank_${SLURM_PROCID}"

MODEL_ARGS=(
--num-layers ${NUM_LAYERS}
--hidden-size ${HIDDEN_SIZE}
--num-attention-heads ${NUM_ATTN_HEADS}
--seq-length ${SEQUENCE_LEN}
--max-position-embeddings ${SEQUENCE_LEN}
--micro-batch-size ${MICRO_BATCH_SIZE}
--global-batch-size ${GLOBAL_BATCH_SIZE}
--lr 0.0005
--train-iters 20
--lr-decay-iters 20
--weight-decay 0.1
--log-interval 5
)

TIMER_ARGS=(
--timing-log-level ${TIMER_LEVEL} # max log level
)

## Runs based on Megatron upstream repo commit c7a1f82
## THIS IS REQUIRED 
# RUN git clone -b codep_settings  https://github.com/schheda1/Megatron-LM.git
# to get the correct versioning info.   Docker image based on NV-PT 24.05-py3 
PYTHONPATH=/pscratch/sd/s/schheda/summer24/sc-Megatron-LM:$PYTHONPATH

python3 -u  pretrain_gpt.py \
	--tensor-model-parallel-size ${TP_SIZE} \
        --pipeline-model-parallel-size ${PP_SIZE} \
	--context-parallel-size ${CP_SIZE} \
	--use-mapping ${MAPPING} \
        --overlap-grad-reduce \
        --overlap-param-gather \
        --use-distributed-optimizer \
        --fp16 \
        --use-flash-attn \
        --distributed-backend nccl \
        ${MODEL_ARGS[@]} \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --data-path ${DATA_PATH} \
        --num-dataset-builder-threads 24 \
        --log-throughput \
        --log-progress \
	${TIMER_ARGS[@]} \
        --split 969,30,1
