set -x

# Input Params:
# 1. num_workers, defaults to 2
# 2. num_gpus_per_worker, defaults to 8
# 3. mp_size, defaults to 8
# 4. dp_size, defaults to 2
# 5. use_checkpointing or not, default to 1

# USE_CHECKPOINTING

bash ds_pretrain_nvidia_acc.sh 2 8 16 1 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=16,DP_SIZE=1
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_acc.sh 2 8 8 2 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=8,DP_SIZE=2
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_acc.sh 2 8 4 4 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=4,DP_SIZE=4
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_acc.sh 2 8 16 1 0 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=16,DP_SIZE=1
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_acc.sh 2 8 8 2 0 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=8,DP_SIZE=2
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_acc.sh 2 8 4 4 0 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=4,DP_SIZE=4
sleep 60
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60
