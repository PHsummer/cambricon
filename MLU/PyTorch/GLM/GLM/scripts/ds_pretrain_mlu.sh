#! /bin/bash

# Set default environment variable
export WIKIPEDIA_DATASET_PATH="/data/cpm/glm/wikipedia_and_bookcorpus"
export PILE_DATASET_PATH="/data/cpm/glm/pile/train"
export LID_176_PATH="/data/cpm/glm/fastText/lid.176.bin"


# Change for multinode config
NUM_WORKERS=1
NUM_MLUS_PER_WORKER=2
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source ${1-"./config/ds_block_base.sh"}
DATESTR=$(date +"%m-%d-%H-%M")

# Run model with deepspeed
run_cmd="deepspeed --master_port ${MASTER_PORT} \
  --num_nodes ${NUM_WORKERS} \
  --num_devs ${NUM_MLUS_PER_WORKER} \
  pretrain_glm.py ${gpt_options} 2>&1 | tee deepspeed-glm-base-log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}
