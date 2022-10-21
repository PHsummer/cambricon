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
  docker exec -it "${CONT_NAME}" bash -c "python config_update.py --config config/st_gcn/kinetics-skeleton/train.yaml --data_path /data/kinetics-skeleton --num_gpus ${device_count} --batch_size ${batch_size} --epochs 1 && python main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
elif [[ ${dev_type} == "mlu"  ]]; then
  if [[ ${device_count} == 1 ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python config_update.py --config config/st_gcn/kinetics-skeleton/train.yaml --data_path /data/kinetics-skeleton --num_gpus ${device_count} --batch_size ${batch_size} --epochs 1 && python main.py recognition -c config/st_gcn/kinetics-skeleton/train.yaml" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  else
    echo "only support single card so far"
  fi
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
