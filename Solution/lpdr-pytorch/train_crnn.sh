CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--nnodes=1 \
--node_rank=0 \
--master_addr="127.5.2.1" \
--master_port=1234 \
train_crnn.py \
--data_path=/workspace/LPDR/Database/DB_Recognition \
--batch_size=64 \
--epochs=100