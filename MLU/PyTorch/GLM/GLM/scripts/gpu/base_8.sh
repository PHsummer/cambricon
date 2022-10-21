#! /bin/bash

# Change for multinode config

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

source ./config/ds_block_base_gpu.sh
DATESTR=$(date +"%m-%d-%H-%M")

run_cmd="python -m torch.distributed.launch \
  --nnodes=${NUM_WORKERS} \
  --nproc_per_node=${NUM_GPUS_PER_WORKER} \
  pretrain_glm.py ${gpt_options} 2>&1 | tee logs/base-log-${DATESTR}.txt"

echo ${run_cmd}
eval ${run_cmd}
