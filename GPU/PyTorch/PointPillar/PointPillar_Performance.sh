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
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
    docker exec -it "${CONT_NAME}"  bash -c "cd tools && sh scripts/dist_train.sh 1 --epochs 1 --batch_size ${batch_size} --cfg_file /workspace/cbgs_pp_multihead.yaml" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    docker exec -it "${CONT_NAME}"  bash -c "cd tools && sh scripts/dist_train.sh ${device_count} --epochs 1 --batch_size ${batch_size}  --cfg_file /workspace/cbgs_pp_multihead.yaml" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
batch_time=$(grep  "b_time"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -1 | cut -d "(" -f 4 | cut -d ")" -f 1)
throughput_all=$(echo "scale=3; ${batch_size}/$batch_time*$device_count"|bc)
echo "total throughput:" $throughput_all

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
echo "network:PointPillar, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} fps, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
