#Set MLU ids
max_mlu_index=$(echo "${device_count}-1"|bc)
mlu_list=$(seq 0 ${max_mlu_index} | xargs -n 16 echo | tr ' ' ',')

# Choose FP32 or FP16
if [[ ${precision} == "FP32" ]]
then 
    amp=""
elif [[ ${precision} == "FP16" ]]
then
    amp="--amp"
fi

# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "  cd imagenet && python main.py /data/imagenet_training --arch  shufflenet_v2_x2_0  -b ${batch_size} --gpu 0 --iters 200 --epoch 1 " 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu"  ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate &&  cd imagenet && python main.py /data/imagenet_training --arch shufflenet_v2_x2_0   -b ${batch_size} --gpu 0 --iters 200 --epoch 1 " 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
else
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "cd imagenet &&  CUDA_VISIBLE_DEVICES=${mlu_list}  python main.py /data/imagenet_training  --arch  shufflenet_v2_x2_0 -b ${batch_size}  --iters 200 --epoch 1 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log   
  elif [[ ${dev_type} == "mlu"  ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && cd imagenet &&  MLU_VISIBLE_DEVICES=${mlu_list}  python main.py /data/imagenet_training  --arch  shufflenet_v2_x2_0  -b ${batch_size}  --iters 200 --epoch 1 --dist-url 'tcp://127.0.0.1:12345' --dist-backend 'cncl' --multiprocessing-distributed --world-size 1 --rank 0" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log   
  fi
fi

# Get performance results
iter_time=$( grep 200/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log  | cut -d "(" -f 2 | head -c 6)
throughput_all=$(echo "${batch_size}/$iter_time"|bc)
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