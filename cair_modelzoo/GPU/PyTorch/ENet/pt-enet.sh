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
docker exec -it "${CONT_NAME}" bash -c "git apply --ignore-space-change --ignore-whitespace --reject modify.patch && python -m torch.distributed.launch --nproc_per_node=${device_count} --nnodes=1 --node_rank=0 main.py -m train --deterministic --batch-size ${batch_size} --epochs 10 --save-dir ./checkpoints --name enet --dataset cityscapes --dataset-dir /data/CityScapes" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log

# Get performance results
throughput=$(grep  "Speed Avg."  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 | cut -d " " -f 3)
throughput_all=$(echo "$throughput*$device_count"|bc)

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
echo "network:ENet, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} images/sec, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log

