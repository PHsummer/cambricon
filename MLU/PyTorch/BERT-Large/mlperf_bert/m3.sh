data=/data/pytorch/datasets/bert_data/

#python -u -m bind_pyt --nsockets_per_node=1 --ncores_per_socket=1 --nproc_per_node=1 \
python -m torch.distributed.launch --nproc_per_node=2 \
	./run_pretraining.py --train_batch_size=20 --learning_rate=4e-4 --opt_lamb_beta_1=0.9 --opt_lamb_beta_2=0.999 \
	--warmup_proportion=0.0 --warmup_steps=0.0 --start_warmup_step=0 --max_steps=200000 --phase2 --max_seq_length=512 --max_predictions_per_seq=76 \
    --input_dir=$data/2048_shards_uncompressed \
    --init_checkpoint=$data/model.ckpt-28252.pt \
	--do_train --skip_checkpoint --train_mlm_accuracy_window_size=0 \
	--target_mlm_accuracy=0.720 --weight_decay_rate=0.01 --max_samples_termination=9000000 \
	--eval_iter_start_samples=150000 --eval_iter_samples=150000 --eval_batch_size=8 \
    --eval_dir=$data/eval_set_uncompressed/ \
	--cache_eval_data --output_dir=./results --fp16 --fused_gelu_bias \
	--dense_seq_output \
	--dwu-num-rs-pg=1 --dwu-num-ar-pg=1 --dwu-num-blocks=1 --gradient_accumulation_steps=1 --log_freq=1 \
    --bert_config_path=$data/bert_config.json \
	--allreduce_post_accumulation_fp16 \
	--use_ddp \
    --ddp_type="native" \
    --use-mlu \
	--unpad \
	--exchange_padding \
	#--bypass_amp \
    #--distributed_lamb \
	#--allreduce_post_accumulation \

