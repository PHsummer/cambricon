#Set MLU ids
max_mlu_index=$(echo "${device_count}-1"|bc)
mlu_list=$(seq 0 ${max_mlu_index} | xargs -n 16 echo | tr ' ' ',')
 
 
# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "python main.py ctdet --exp_id coco_dla_2x --batch_size ${batch_size} --master_batch 9 --lr 5e-4 --gpus 0 --num_workers 16 --num_epochs 1 --epoch_iter 100 --lr_step 180,210" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu"  ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python main.py ctdet --exp_id coco_dla_2x --batch_size ${batch_size} --master_batch 9 --lr 5e-4 --gpus 0 --num_workers 16 --num_epochs 1 --epoch_iter 100 --lr_step 180,210" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
else
  echo "only support single card so far" 
fi
 
 
# Get performance results
iter_time=$( grep -n "Net " ${LOG_DIR}/${model_name}_${DATESTAMP}.log | cut -d "N" -f 102 | cut -d " " -f 2 | cut -d "s" -f 1 )
throughput=$(echo "${batch_size}/$iter_time"|bc)
throughput_all=$(echo "$device_count*$throughput"|bc)
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