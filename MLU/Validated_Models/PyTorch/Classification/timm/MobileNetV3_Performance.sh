#Set MLU ids
max_mlu_index=$(echo "${device_count}-1"|bc)
mlu_list=$(seq 0 ${max_mlu_index} | xargs -n 16 echo | tr ' ' ',')


# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "python train.py /data/imagenet_training --sched cosine --epochs 1 --cooldown-epochs 0 --batch-size ${batch_size} --iters 300 --model mobilenetv3_large_075" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu"  ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python train.py /data/imagenet_training --sched cosine --epochs 1 --cooldown-epochs 0 --batch-size ${batch_size} --iters 300 --model mobilenetv3_large_075" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
else
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "python -m torch.distributed.launch --nproc_per_node=${device_count} train.py /data/imagenet_training --sched cosine --epochs 1 --cooldown-epochs 0 --batch-size ${batch_size} --iters 300 --model mobilenetv3_large_075 --epochs 1" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu"  ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python -m torch.distributed.launch --nproc_per_node=${device_count} train.py /data/imagenet_training --sched cosine --epochs 1 --cooldown-epochs 0 --batch-size ${batch_size} --iters 300 --model mobilenetv3_large_075 --epochs 1" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
fi

# Get performance results
#throughput=$( grep 250/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log | head -1 | cut -d ")"  -f 3 | tail -c 9 | head -c 6)
throughput=$( grep 250/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log | head -n 1 | cut -d ")" -f 3 | cut -d "," -f 3 | tr -cd ".0-9")
throughput_all=$(echo "$throughput"|bc)
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
