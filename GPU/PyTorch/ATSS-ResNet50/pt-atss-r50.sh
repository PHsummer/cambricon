# choose GPU type
if [[ $gpu =~ "A100" ]]
then
    DGXSYSTEM="DGXA100"
elif [[ $gpu =~ "V100" ]]
then
    DGXSYSTEM="DGX1V"   
fi 

# for TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi

# Run the model
docker exec -it "${CONT_NAME}" bash -c "git apply --ignore-space-change --ignore-whitespace --reject modify.patch && ./tools/dist_train.sh configs/atss/atss_r50_fpn_1x_coco.py ${device_count} /data/COCO17/ ${batch_size} 1" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log

# Get performance results
throughput=$(grep  ", time: "  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 | cut -d " " -f 14 | cut -d "," -f 1)
throughput_all=$(echo "scale=2;${batch_size}*${device_count}/${throughput}"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"
       
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="single"
else
  dpf_mode="DDP"
fi

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} images/sec, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
