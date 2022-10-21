#! /bin/bash

script_path=$(realpath $BASH_SOURCE)
script_dir=$(dirname $script_path)

config_json="$script_dir/config_block_10B_acc_ac${USE_CHECKPOINTING}.json"
gpt_options=" \
       --block-lm \
       --task-mask \
       --bert-prob 0.5 \
       --gap-sentence-prob 0.3 \
       --avg-block-length 3 \
       --gpt-min-ratio 0.25 \
       --block-mask-prob 0.1 \
       --short-seq-prob 0.02 \
       --experiment-name blocklm-10b \
       --model-parallel-size ${MP_SIZE} \
       --num-layers 48 \
       --hidden-size 4096 \
       --num-attention-heads 64 \
       --seq-length 512 \
       --max-position-embeddings 1024 \
       --train-iters 1000 \
       --log-interval 10 \
       --train-data pile \
       --filter-english \
       --loader-scatter ${LOADER_SCATTER} \
       --tokenizer-type GPT2BPETokenizer \
       --split 1000,0,0 \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --lr-decay-ratio 0.1 \
       --lr-decay-iters 175000 \
       --warmup 0.04 \
       --fp16 \
"
if [ ${USE_CHECKPOINTING} -eq 1 ]; then
  gpt_options+="${gpt_options}
                 --checkpoint-activations \
                 --deepspeed-activation-checkpointing \
                 "
fi
gpt_options="${gpt_options}
               --deepspeed \
               --deepspeed_config ${config_json} \
"
       # --save /dataset/fd5061f6/english_data/checkpoints \
       # --train-data pile cc-news \
       # --resume-dataloader \
       # --eval-interval 1000 \
       # --save-interval 2000 \
