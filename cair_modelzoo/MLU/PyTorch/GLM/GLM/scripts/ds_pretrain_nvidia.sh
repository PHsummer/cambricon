#! /bin/bash

set -e
set -x
# Change for multinode config

# Input Params:
# 1. num_workers, defaults to 2
# 2. num_gpus_per_worker, defaults to 8
# 3. mp_size, defaults to 8
# 4. relative_ds_config_file_path, defaults to config/ds_block_10B.sh

NUM_WORKERS=${1:-2}
NUM_GPUS_PER_WORKER=${2:-8}
MP_SIZE=${3:-8}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# MASTER_PORT="6006"

source ${4:-"config/ds_block_10B.sh"}
DATESTR=$(date +"%m-%d-%H-%M")

# OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=2"
OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="/home/zhangchi1/glm/hostfile.txt"

if [ ! -d logs ]; then
  mkdir logs
fi

#For socket if no IB
#export OMPI_MCA_btl_tcp_if_include="eth0"
#export NCCL_SOCKET_IFNAME=eth0

# run_cmd="${OPTIONS_NCCL} deepspeed \
run_cmd="deepspeed \
           --master_port ${MASTER_PORT} \
           --num_nodes ${NUM_WORKERS} \
           --num_gpus ${NUM_GPUS_PER_WORKER} \
           --hostfile ${HOST_FILE_PATH} \
           pretrain_glm.py \
           ${gpt_options} 2>&1 | tee logs/log-${DATESTR}.txt"
echo ${run_cmd}
eval ${run_cmd}
