#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
DATASET=$3
BATCH_SIZE=$4
EPOCHS=$5
ITERS=$6
INTERVAL=$7
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python $(dirname "$0")/config_update.py \
--config=$CONFIG \
--data_path=$DATASET \
--batch_size=$BATCH_SIZE \
--epochs=$EPOCHS \
--iters=$ITERS \
--interval=$INTERVAL

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --deterministic \
    --launcher pytorch
    --no-validate
