#data=/data/pytorch/datasets/bert_data/
data=/data/bert_large_uncased/bert_data

python -m torch.distributed.launch --nproc_per_node=4 \
    ./run_pretraining.py \
    --train_batch_size=4 \
    --learning_rate=4e-5 \
    --opt_lamb_beta_1=0.83 \
    --opt_lamb_beta_2=0.925 \
    --warmup_proportion=0.0 \
    --warmup_steps=0 \
    --start_warmup_step=0 \
    --phase2 \
    --max_seq_length=512 \
    --max_predictions_per_seq=76 \
    --input_dir=$data/2048_shards_uncompressed \
    --init_checkpoint=$data/model.ckpt-28252.pt \
    --do_train \
    --skip_checkpoint \
    --train_mlm_accuracy_window_size=0 \
    --target_mlm_accuracy=0.720 \
    --weight_decay_rate=0.01 \
    --max_samples_termination=4500000 \
    --eval_iter_start_samples=100000 \
    --eval_iter_samples=100000 \
    --eval_batch_size=4 \
    --eval_dir=$data/eval_set_uncompressed/ \
    --cache_eval_data \
    --output_dir=./results \
    --dense_seq_output \
    --dwu-num-rs-pg=1 \
    --dwu-num-ar-pg=1 \
    --dwu-num-blocks=1 \
    --gradient_accumulation_steps=1 \
    --log_freq=100 \
    --bert_config_path=$data/bert_config.json \
    --allreduce_post_accumulation_fp16 \
    --use_ddp \
    --fp16 \
    --max_step=1000000000 \
    --ddp_type="native"
