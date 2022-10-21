#!/bin/bash
set -e

CUR_DIR=$(cd $(dirname $0);pwd)
WORK_DIR=$(cd ${CUR_DIR}/../;pwd)

export MASTER_ADDR='127.0.0.2'
export MASTER_PORT=23456

pushd $WORK_DIR

pip install -r requirements.txt

output_dir="./outputTest/"
dataset=${PYTORCH_TRAIN_DATASET}/TTS/
mkdir -p ${output_dir}

cmd="python -m torch.distributed.run --nproc_per_node=2  train.py -m Tacotron2 -o \
    ${output_dir}  -lr 1e-3 --epochs 1501 -bs 128  --weight-decay 1e-6 --grad-clip-thresh 1.0 \
    --dist-url tcp://$MASTER_ADDR:$MASTER_PORT --pyamp --log-file nvlog.json --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 -d ${dataset} --use-mlu --cudnn-enabled --dist-backend cncl"
echo ${cmd}
eval ${cmd}
popd



