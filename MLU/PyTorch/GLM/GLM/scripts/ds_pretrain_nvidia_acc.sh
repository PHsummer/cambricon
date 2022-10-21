#! /bin/bash

set -e
set -x
# Change for multinode config

# Input Params:
# 1. num_workers, defaults to 2
# 2. num_gpus_per_worker, defaults to 8
# 3. mp_size, defaults to 8
# 4. dp_size, defaults to 2
# 5. use_checkpointing or not, default to 1
# 6. overlap_comm or not, default to 0

NUM_WORKERS=${1:-2}
NUM_GPUS_PER_WORKER=${2:-8}
MP_SIZE=${3:-8}
DP_SIZE=${4:-2}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
USE_CHECKPOINTING=${5:-1}
OVERLAP_COMM=${6:-1}

LOADER_SCATTER=${NUM_WORKERS}
if [ ${NUM_WORKERS} -eq 2 -a ${DP_SIZE} -eq 1 ]; then
  LOADER_SCATTER=1
fi

source "config/ds_block_10B_acc.sh"

# DATESTR=$(date +"%m-%d-%H-%M")
DATESTR=$(date +"%H-%M")

WORK_DIR=$(cd $(dirname $0); pwd)
HOST_FILE_PATH="${WORK_DIR}/hostfile.txt"

# you should set this in .deepspeed_env
# PILE_DATASET_PATH="/algo/datasets_training/pile/train_00/"
# LID_176_PATH="/algo/datasets_training/pile/glm/fastText/lid.176.bin"

if [ ! -d acc_logs ]; then
  mkdir acc_logs
fi

#For socket if no IB
# export NCCL_SOCKET_IFNAME=eth0

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# run_cmd="${OPTIONS_NCCL} deepspeed \
run_cmd="deepspeed \
           --master_port ${MASTER_PORT} \
           --num_nodes ${NUM_WORKERS} \
           --num_gpus ${NUM_GPUS_PER_WORKER} \
           --hostfile ${HOST_FILE_PATH} \
           pretrain_glm.py \
           ${gpt_options} 2>&1 | tee acc_logs/log-${DATESTR}_${NUM_WORKERS}nodes_mp${MP_SIZE}_dp${DP_SIZE}_ckpt${USE_CHECKPOINTING}.txt"
echo ${run_cmd}
eval ${run_cmd}
