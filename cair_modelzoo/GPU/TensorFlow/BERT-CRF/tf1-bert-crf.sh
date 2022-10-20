# for TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi
# Run the model
if [[ $device_count == 1 ]] && [[ $precision != "FP16" ]]
then 
  docker exec -it "${CONT_NAME}" bash -c "python BERT_NER.py --task_name="NER" --do_lower_case=False --crf=True --do_train=True --do_eval=False --do_predict=False --data_dir=data --vocab_file=/data/tensorflow/datasets/google_bert/cased_L-12_H-768_A-12/vocab.txt --bert_config_file=/data/tensorflow/datasets/google_bert/cased_L-12_H-768_A-12/bert_config.json --init_checkpoint=/data/tensorflow/datasets/google_bert/cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=${batch_size} --learning_rate=2e-5 --num_train_epochs=1.0 --output_dir=./output/result_dir" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
else
  echo "Multi-GPUs is not supported, and FP16 training is not supported now!"
  return 1
  break
fi

# Get performance results
throughput=$(grep "INFO:tensorflow:examples/sec:" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n 5 | awk '{print $2}' | awk '{sum+=$1}END{print sum/5}')
       
# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"

# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
fi

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput} seqs/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
