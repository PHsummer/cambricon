# Choose GPU type
# pass

# For TF32 case
if [[ $gpu =~ "A100" || $gpu =~ "3090" ]] && [[ $precision == "FP32" ]]
then
    export NVIDIA_TF32_OVERRIDE=0
    echo -e "\033[31m Training with FP32 instead of TF32! \033[0m"
fi

# Set batch size
batch_size=$(echo "${batch_size}*${device_count}"|bc)

# Run the model
if [[ $device_count == 1 ]] && [[ $precision == "AMP" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "python torchvision_main.py -a mobilenet_v2 --batch-size ${batch_size} --epochs 1 --print-freq 10 --gpu 0 -j 12 --iters 500 --amp --no-val --no-save /data" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ $device_count == 1 ]] && [[ $precision == "FP32" || $precision == "TF32" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "python torchvision_main.py -a mobilenet_v2 --batch-size ${batch_size} --epochs 1 --print-freq 10 --gpu 0 -j 12 --iters 500 --no-val --no-save /data" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ $device_count > 1 ]] && [[ $precision == "AMP" ]]
then 
    max_gpu_index=$(echo "${device_count}-1"|bc)
    gpu_list=$(seq 0 ${max_gpu_index} | xargs -n 16 echo | tr ' ' ',')
    docker exec -it "${CONT_NAME}" bash -c "CUDA_VISIBLE_DEVICES=${gpu_list} python torchvision_main.py -a mobilenet_v2 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --batch-size ${batch_size} --epochs 1 --print-freq 10 -j 12 --iters 500 --amp --no-val --no-save /data" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ $device_count > 1 ]] && [[ $precision == "FP32" || $precision == "TF32" ]]
then 
    max_gpu_index=$(echo "${device_count}-1"|bc)
    gpu_list=$(seq 0 ${max_gpu_index} | xargs -n 16 echo | tr ' ' ',')
    docker exec -it "${CONT_NAME}" bash -c "CUDA_VISIBLE_DEVICES=${gpu_list} python torchvision_main.py -a mobilenet_v2 --dist-url 'tcp://127.0.0.1:8800' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --batch-size ${batch_size} --epochs 1 --print-freq 10 --gpu 0 -j 12 --iters 500 --no-val --no-save /data" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
fi

# Get performance results
throughput=$(grep "FPS" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n 10 | awk '{print $20}' | awk '{sum+=$1}END{print sum/10}')
throughput_all=$(echo "$throughput*$device_count"|bc)

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
batch_size=$(echo "${batch_size}/${device_count}"|bc)

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} img/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
