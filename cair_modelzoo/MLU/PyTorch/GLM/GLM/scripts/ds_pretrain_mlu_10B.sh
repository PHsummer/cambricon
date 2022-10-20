#! /bin/bash

# Set default environment variable
export PILE_DATASET_PATH="/data/cpm/glm/pile/train"
export LID_176_PATH="/data/cpm/glm/fastText/lid.176.bin"

# Change for multinode config
NUM_WORKERS=2
NUM_MLUS_PER_WORKER=16
MP_SIZE=8
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOSTFILE="hostfile.txt"
MASTER_ADDR="set_master_addr"


source ${1-"./config/ds_block_10B.sh"}
DATESTR=$(date +"%m-%d-%H-%M")

run_cmd="deepspeed --master_port ${MASTER_PORT} \
  --num_nodes ${NUM_WORKERS} \
  --num_devs ${NUM_MLUS_PER_WORKER} \
  --master_addr $MASTER_ADDR \
  --hostfile=$HOSTFILE \
  --launcher="pdsh" \
  pretrain_glm.py ${gpt_options} 2>&1 | tee deepspeed-log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}
