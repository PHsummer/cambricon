# Run experiments
docker exec -it "${CONT_NAME}" bash -c "bash run_ner.sh" 2>&1 | tee ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log
echo -e "\033[32m Run experiments done! \033[0m"

# Get convergence results
#throughput=$(grep "INFO:tensorflow:examples/sec:" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n 5 | awk '{print $2}' | awk '{sum+=$1}END{print sum/5}')
accuracy=$(grep "accuracy:" ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log | awk '{print $2}' | sed 's/\;//g')

# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
fi

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="98.16%"

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, accuracy:$accuracy, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
