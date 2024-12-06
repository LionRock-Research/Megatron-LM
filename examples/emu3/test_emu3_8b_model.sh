#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

HF_CKPT_PATH="/data/models/Emu3-Gen"
CHECKPOINT_PATH="/data/models/Emu3-Gen-Megatron"
TENSORBOARD_LOGS_PATH="./logs"
# VOCAB_FILE=$3 #<Specify path to file>/gpt2-vocab.json
# MERGE_FILE=$4 #<Specify path to file>/gpt2-merges.txt
# DATA_PATH="./"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 32
    --hidden-size 4096
    --num-attention-heads 32 
    --num-query-groups 8
    --seq-length 2048 
    --max-position-embeddings 2048
    --vocab-size 184622
    --rotary-base 1000000
    --use-rotary-position-embeddings
    --normalization "RMSNorm"
    --attention-softmax-in-fp32
    --ffn-hidden-size 14336
    --make-vocab-size-divisible-by 2
    --swiglu
    --disable-bias-linear
)

TOKENIZER_ARGS=(
    --tokenizer-type "HuggingFaceTokenizer"
    --tokenizer-model $HF_CKPT_PATH
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 4 
    --train-iters 5000
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --bf16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

DATA_ARGS=(
#    --data-path $DATA_PATH 
#    --vocab-file $VOCAB_FILE 
#    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000 
#    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

EXTRA_ARGS=(
    --transformer-impl "transformer_engine"
    --mock-data
)

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${EXTRA_ARGS[@]}
 