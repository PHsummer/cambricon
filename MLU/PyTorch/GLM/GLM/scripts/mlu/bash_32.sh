#! /bin/bash

# Set default environment variable
export WIKIPEDIA_DATASET_PATH="/data/glm/wikipedia_and_bookcorpus"

NUM_WORKERS=2
NUM_MLUS_PER_WORKER=16
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source ./config/ds_block_base_mlu_acc.sh
DATESTR=$(date +"%m-%d-%H-%M")

run_cmd="deepspeed --master_port ${MASTER_PORT} \
  --num_nodes ${NUM_WORKERS} \
  --num_devs ${NUM_MLUS_PER_WORKER} \
  --master_addr 10.0.1.6 \
  --hostfile="./hostfile" \
  --launcher="pdsh" \
  pretrain_glm.py ${gpt_options} 2>&1 | tee logs/deepspeed-log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}

