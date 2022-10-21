#Set MLU ids
max_mlu_index=$(echo "${device_count}-1"|bc)
mlu_list=$(seq 0 ${max_mlu_index} | xargs -n 16 echo | tr ' ' ',')


# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "python run_bert_triple_classifier.py --task_name kg --do_train --data_dir ./data/WN11 --bert_model /model/bert-base-uncased --max_seq_length 20 --train_batch_size ${batch_size} --learning_rate 5e-5 --num_train_epochs 1.0 --iters 20 --output_dir ./output_WN11/  --gradient_accumulation_steps 1 --eval_batch_size ${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python run_bert_triple_classifier.py --task_name kg --do_train --data_dir ./data/WN11 --bert_model /model/bert-base-uncased --max_seq_length 20 --train_batch_size ${batch_size} --learning_rate 5e-5 --num_train_epochs 1.0 --iters 20 --output_dir ./output_WN11/  --gradient_accumulation_steps 1 --eval_batch_size ${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
else
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "python -m torch.distributed.launch --nproc_per_node=${device_count} run_bert_triple_classifier.py --task_name kg --do_train  --do_eval --do_predict --data_dir ./data/WN11 --bert_model /model/bert-base-uncased --max_seq_length 20 --train_batch_size ${batch_size} --learning_rate 5e-5 --num_train_epochs 1.0 --iters 20 --output_dir ./output_WN11/  --gradient_accumulation_steps 1 --eval_batch_size ${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
  elif [[ ${dev_type} == "mlu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "source /torch/venv3/pytorch/bin/activate && python -m torch.distributed.launch --nproc_per_node=${device_count} run_bert_triple_classifier.py --task_name kg --do_train  --do_eval --do_predict --data_dir ./data/WN11 --bert_model /model/bert-base-uncased --max_seq_length 20 --train_batch_size ${batch_size} --learning_rate 5e-5 --num_train_epochs 1.0 --iters 20 --output_dir ./output_WN11/  --gradient_accumulation_steps 1 --eval_batch_size ${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
  fi
fi


# Get performance results
its=$( grep 20/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log |cut -d "," -f 3|grep -v "^$"|head -1|tr -cd ".0-9")
throughput_all=$(echo "scale=2; $batch_size/$its*$device_count"|bc)
if [[ -z $throughput_all ]]; then
  echo "Failed to dump throughput"
  exit 1
fi
echo "========================================="
echo "its:", $its
echo "throughput_all:", $throughput_all
       
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
else
  dpf_mode="DDP"
fi


# Write benchmark log into a file
if [[ ${dev_type} == "gpu" ]]; then
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} samples/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CUDA:${cuda_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
elif [[ ${dev_type} == "mlu"  ]]; then
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} samples/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CTR:${ctr_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
fi
