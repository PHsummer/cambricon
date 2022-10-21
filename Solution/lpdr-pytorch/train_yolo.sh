#！/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_addr="127.0.0.1" \
--master_port=1234 \
train_yolo.py \
--data_path=/workspace/LPDR/Database/DB_Detection \
--batch_size=16 \
--epochs=300