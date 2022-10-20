#Set MLU ids
max_mlu_index=$(echo "${device_count}-1"|bc)
mlu_list=$(seq 0 ${max_mlu_index} | xargs -n 16 echo | tr ' ' ',')

# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "export CUDA_VISIBLE_DEVICES=0 && cp /model/cached_train_chinese-macbert-base_512 ./ && python run_squad.py --model_type roberta --model_name_or_path /model/chinese-macbert-base --do_train --do_lower_case --train_file /data/DRCD/DRCD_training.json --predict_file /data/DRCD/DRCD_dev.json --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${batch_size} --learning_rate 3e-5 --num_train_epochs 1.0 --iters 30 --max_query_length 222 --max_answer_length 118 --max_seq_length 512 --doc_stride 128 --output_dir ./outputs_bert --overwrite_output_dir --gradient_accumulation_steps 1 --device gpu" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  elif [[ ${dev_type} == "mlu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "export MLU_VISIBLE_DEVICES=0 && cp /model/cached_train_chinese-macbert-base_512 ./ && source /torch/venv3/pytorch/bin/activate && python run_squad.py --model_type roberta --model_name_or_path /model/chinese-macbert-base --do_train --do_lower_case --train_file /data/DRCD/DRCD_training.json --predict_file /data/DRCD/DRCD_dev.json --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${batch_size} --learning_rate 3e-5 --num_train_epochs 1.0 --iters 30 --max_query_length 222 --max_answer_length 118 --max_seq_length 512 --doc_stride 128 --output_dir ./outputs_bert --overwrite_output_dir --gradient_accumulation_steps 1 --device mlu" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
  fi
else
  if [[ ${dev_type} == "gpu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "cp /model/cached_train_chinese-macbert-base_512 ./ && python -m torch.distributed.launch --nproc_per_node=${device_count} run_squad.py --model_type roberta --model_name_or_path /model/chinese-macbert-base --do_train --do_lower_case --train_file /data/DRCD/DRCD_training.json --predict_file /data/DRCD/DRCD_dev.json --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${batch_size} --learning_rate 3e-5 --num_train_epochs 1.0 --iters 30 --max_query_length 222 --max_answer_length 118 --max_seq_length 512 --doc_stride 128 --output_dir ./outputs_bert --overwrite_output_dir --gradient_accumulation_steps 1 --device gpu" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
  elif [[ ${dev_type} == "mlu" ]]; then
    docker exec -it "${CONT_NAME}" bash -c "cp /model/cached_train_chinese-macbert-base_512 ./ && source /torch/venv3/pytorch/bin/activate && python -m torch.distributed.launch --nproc_per_node=${device_count} run_squad.py --model_type roberta --model_name_or_path /model/chinese-macbert-base --do_train --do_lower_case --train_file /data/DRCD/DRCD_training.json --predict_file /data/DRCD/DRCD_dev.json --per_gpu_train_batch_size ${batch_size} --per_gpu_eval_batch_size ${batch_size} --learning_rate 3e-5 --num_train_epochs 1.0 --iters 30 --max_query_length 222 --max_answer_length 118 --max_seq_length 512 --doc_stride 128 --output_dir ./outputs_bert --overwrite_output_dir --gradient_accumulation_steps 1 --device mlu" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
  fi
fi

# Get performance results
#its=$( grep 20/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log |cut -d "," -f 3|grep -v "^$"|head -1|tr -cd ".0-9")
#throughput_all=$(echo "scale=2; $batch_size/$its*$device_count"|bc)
perf_num=$( grep 30/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log | cut -d "," -f 2 | cut -d "]" -f 1 | tr -cd ".0-9" )
perf_unit=$( grep 30/ ${LOG_DIR}/${model_name}_${DATESTAMP}.log | cut -d "," -f 2| cut -d "]" -f 1 | rev | head -c 1 )
if [[ $perf_unit = "s" ]]; then #unit is it/s
    throughput_all=$(echo "scale=2; $batch_size*$perf_num*$device_count"|bc)
else #unit is s/it
    throughput_all=$(echo "scale=2; $batch_size/$perf_num*$device_count"|bc)
fi
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
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} samples/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CUDA:${cuda_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
elif [[ ${dev_type} == "mlu"  ]]; then
  echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} samples/s, device:${dev_name}, dataset:${DATASETS}, PCIE:${pcie}, driver:${driver}, CTR:${ctr_version}" >> ${RESULT_DIR}/${BENCHMARK_LOG}
fi