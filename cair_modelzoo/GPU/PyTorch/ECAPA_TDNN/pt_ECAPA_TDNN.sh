# A demo for PyTorch ResNet50v1.5 single card training, batch_size is fixed at 112 in source code.

# Choose GPU type
if [[ $gpu =~ "A100" ]]
then
    DGXSYSTEM="DGXA100"
elif [[ $gpu =~ "V100" ]]
then
    DGXSYSTEM="DGX1V"   
fi 

# For TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi

# precision
if [[ $precision != "FP32" ]]
then
    echo -e "Precision is not FP32, please set!"
    exit 0
fi

# Set batch size
if [[ ${batch_size} != "32" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "sed -i 's|batch_size: 32|batch_size: ${batch_size}|g' ./train_ecapa_tdnn.yaml" 
    echo -e "The batch size has been set to ${batch_size}"
fi



# Run the model
if [[ $device_count == 1 ]]
then
    docker exec -it "${CONT_NAME}" bash -c "git apply --ignore-space-change --ignore-whitespace --reject modify.patch && python train_speaker_embeddings.py ./train_ecapa_tdnn.yaml --data_folder /data --data_parallel_backend" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    docker exec -it "${CONT_NAME}" bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && git apply --ignore-space-change --ignore-whitespace --reject modify.patch && python -m torch.distributed.launch --nproc_per_node=8 train_speaker_embeddings.py ./train_ecapa_tdnn.yaml --data_folder /data --distributed_launch --distributed_backend='nccl'" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
throughput=$(grep  "Speed Avg."  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 | cut -d " " -f 3)
# throughput_all=$(echo "$throughput*$device_count"|bc)
throughput_all=$(echo "$throughput"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"
     
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="single"
else
  dpf_mode="DDP"
fi

# Write benchmark log into a file
echo "network:ECAPA-TDNN, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} samples/sec, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
