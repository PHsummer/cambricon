#! /bin/bash

# Change for multinode config

NUM_WORKERS=2
NUM_GPUS_PER_WORKER=8
#${1-1}
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source ./config/ds_block_base_gpu.sh
DATESTR=$(date +"%m-%d-%H-%M")

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=1 NCCL_NET_GDR_LEVEL=2"
HOST_FILE_PATH="/glm/hostfile"

export OMPI_MCA_btl_tcl_if_include="eth0"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=info
export NCCL_IB_DISABLE=1


run_cmd="deepspeed --master_port ${MASTER_PORT} \
  --num_nodes ${NUM_WORKERS} \
  --num_gpus ${NUM_GPUS_PER_WORKER} \
  --hostfile ${HOST_FILE_PATH} \
  pretrain_glm.py ${gpt_options} 2>&1 | tee logs/base-log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}
