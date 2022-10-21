set -x

# Input Params:
# 1. num_workers, defaults to 2
# 2. num_gpus_per_worker, defaults to 8
# 3. mp_size, defaults to 8
# 4. dp_size, defaults to 2
# 5. use_checkpointing or not, default to 1

# USE_CHECKPOINTING

# RUN OK
# WITHOUT_OVERLAP_COMM

bash ds_pretrain_nvidia_perf.sh 2 8 16 1 1 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=16,DP_SIZE=1
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_16_1_1_0
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_perf.sh 2 8 8 2 1 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=8,DP_SIZE=2
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_8_2_1_0
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_perf.sh 2 8 4 4 1 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=4,DP_SIZE=4
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_4_4_1_0
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

# WITH_OVERLAP_COMM
 
bash ds_pretrain_nvidia_perf.sh 2 8 16 1 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=16,DP_SIZE=1
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_16_1_1_1
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_perf.sh 2 8 8 2 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=8,DP_SIZE=2
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_8_2_1_1
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60

bash ds_pretrain_nvidia_perf.sh 2 8 4 4 1 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=4,DP_SIZE=4
mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_4_4_1_1
ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
sleep 60


# RUN NOT OK
# # bash ds_pretrain_nvidia_perf.sh 2 8 2 8 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=2,DP_SIZE=8
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_2_8_1
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
 
# # bash ds_pretrain_nvidia_perf.sh 2 8 1 16 1 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=1,DP_SIZE=16
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_1_16_1
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# # bash ds_pretrain_nvidia_perf.sh 1 8 8 1 1 # NUM_NODES=1,WORLD_SIZE=8,MP_SIZE=8,DP_SIZE=1
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_8_8_1_1
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# # bash ds_pretrain_nvidia_perf.sh 1 4 4 1 1 # NUM_NODES=1,WORLD_SIZE=4,MP_SIZE=4,DP_SIZE=1
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_4_4_1_1
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
 
# # bash ds_pretrain_nvidia_perf.sh 1 8 4 2 1 # NUM_NODES=1,WORLD_SIZE=8,MP_SIZE=4,DP_SIZE=2
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_8_4_2_1
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# NO_USE_CHECKPOINTING
# ALL RUN NOT OK

# # bash ds_pretrain_nvidia_perf.sh 2 8 8 2 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=8,DP_SIZE=2
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_8_2_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# # bash ds_pretrain_nvidia_perf.sh 2 8 4 4 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=4,DP_SIZE=4
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_4_4_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# # bash ds_pretrain_nvidia_perf.sh 2 8 2 8 0 # NUM_NODES=2,WORLD_SIZE=16,MP_SIZE=2,DP_SIZE=8
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_2_8_2_8_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
 
# # bash ds_pretrain_nvidia_perf.sh 1 8 8 1 0 # NUM_NODES=1,WORLD_SIZE=8,MP_SIZE=8,DP_SIZE=1
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_8_8_1_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9

# # bash ds_pretrain_nvidia_perf.sh 1 4 4 1 0 # NUM_NODES=1,WORLD_SIZE=4,MP_SIZE=4,DP_SIZE=1
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_4_4_1_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
 
# # bash ds_pretrain_nvidia_perf.sh 1 8 4 2 0 # NUM_NODES=1,WORLD_SIZE=8,MP_SIZE=4,DP_SIZE=2
# # mv algo_glm_dsp_nvprof_xxlarge algo_glm_dsp_nvprof_xxlarge_1_8_4_2_0
# # ps aux | grep pretrain_glm | awk '{print $2}' | xargs kill -9
