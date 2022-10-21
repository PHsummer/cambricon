# Choose FP32 or FP16
if [[ ${precision} == "FP32" ]]
then
    amp=""
elif [[ ${precision} == "FP16" ]]
then
    amp="--amp"
fi
 
# Run the model
if [[ ${dev_type} == "gpu" ]]; then
  docker exec -it "${CONT_NAME}" bash -c "python src/train.py -b ${batch_size} -d /data/GOTURN/ -n 20" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ ${dev_type} == "mlu"  ]]; then
  docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python src/train.py -b ${batch_size} -d /data/GOTURN/ -n 20" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
fi
 
# Get performance results
throughput=$(grep  "Speed Avg."  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 | cut -d " " -f 3)
throughput_all=$(echo "$throughput*$device_count"|bc)
if [[ -z $throughput_all ]]; then
  echo "Failed to dump throughput"
  exit 1
fi
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
else
  dpf_mode="DDP"
fi
 
# Write benchmark log into a file
if [[ ${dev_type} == "gpu" ]]; then
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} img/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CUDA:${cuda_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
elif [[ ${dev_type} == "mlu"  ]]; then
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} img/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CTR:${ctr_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
fi