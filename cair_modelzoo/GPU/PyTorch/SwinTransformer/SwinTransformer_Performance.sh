# Choose GPU type
# pass

# For TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    export NVIDIA_TF32_OVERRIDE=0
    echo -e "\033[31m Training with FP32 instead of TF32! \033[0m"
fi



# Run the model
if [[ $precision == "AMP" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "python -m torch.distributed.launch --nproc_per_node ${device_count} --master_port 1234 main.py --cfg configs/swin/${model_name}.yaml  --data-path /data --batch-size ${batch_size} --amp-opt-level O1 --iter 300" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ $precision == "FP32" || $precision == "TF32" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "python -m torch.distributed.launch --nproc_per_node ${device_count} --master_port 1234 main.py --cfg configs/swin/${model_name}.yaml  --data-path /data --batch-size ${batch_size} --amp-opt-level O0 --iter 300" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
fi

# Get performance results
iter_time=$(grep "300/" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | cut -d "(" -f 3 | cut -d ")" -f 1)
throughput_all=$(echo "${batch_size}/$iter_time*$device_count"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"
       
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
elif [[ $device_count > 1 ]]
then
  dpf_mode="DDP"
fi

# Data precision
if [[ $precision == "AMP" ]]
then
    precision="FP16"
fi

# Reset batch size
#batch_size=$(echo "${batch_size}*${device_count}"|bc)

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} img/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
