
# for TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi

# Run the model
if [[ $precision != "FP16" ]]
then
    max_gpu_index=$(echo "${device_count}-1"|bc)
    gpu_list=$(seq 0 ${max_gpu_index} | xargs -n 16 echo | tr ' ' ',')
    docker exec -it "${CONT_NAME}" bash -c "sed -i '66s/$/\n        Xv_valid_ = None/g' main.py && sed -i '71s/$/\n        break/g' main.py && sed -i '69s/$/\n        print(\"train samples:\",len(y_train_))/g' main.py && sed -i "s/1024,/${batch_size},/g" main.py && rm -rf ./data && ln -s /data/tensorflow/datasets/DeepFM/data ./ && sed -i '156s/^/#/g' main.py && sed -i '162s/^/#/g' main.py && CUDA_VISIBLE_DEVICES=${gpu_list} python main.py" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    echo "FP16 training is not supported!"
fi

# Get performance results
sample_per_epoch=$(grep  "train samples:"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | awk '{print$3}')
time_per_epoch=$(grep "train-result=" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n 9 | awk '{print$3}' | sed 's/\[//g' | awk '{sum+=$1}END{print sum/9}')
throughput=$(echo "$sample_per_epoch/$time_per_epoch" | tr -d $'\r' | bc)
throughput_all=$(echo "$throughput*$device_count"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"

# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
fi

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} items/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
