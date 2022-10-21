#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

gpt_options=" \
       --block-lm \
       --fp16 \
       --bert-prob 1.0 \
       --experiment-name blocklm-blank \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 12 \
       --hidden-size 768 \
       --num-attention-heads 12 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --save /glm/ \
       --batch-size 16 \
       --train-iters 150000 \
       --resume-dataloader \
       --train-data bert-base \
       --tokenizer-type BertWordPieceTokenizer \
       --tokenizer-model-type bert-base-uncased \
       --split 949,50,1 \
       --distributed-backend cncl \
       --lr-decay-style cosine \
       --lr-decay-iters 120000 \
       --lr-decay-ratio 0.05 \
       --warmup .05 \
"
