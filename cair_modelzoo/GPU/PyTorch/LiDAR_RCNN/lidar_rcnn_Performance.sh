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
if [[ $device_count == 1 ]]
then
    docker exec -it "${CONT_NAME}"  bash -c "python -m torch.distributed.launch --nproc_per_node=1 tools/train.py --cfg config/lidar_rcnn_all_cls_2x.yaml --name lidar_rcnn " 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    docker exec -it "${CONT_NAME}"  bash -c "python -m torch.distributed.launch --nproc_per_node=8 tools/train.py --cfg config/lidar_rcnn_all_cls.yaml --name lidar_rcnn " 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
if [[ $device_count == 1 ]]
then
    throughput_all=$(grep "8000/28684"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 |  cut -d "," -f 3 | cut -d " " -f 3 )
else
    throughput_all=$(grep "2000/3585"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 |  cut -d "," -f 3 | cut -d " " -f 3 )
fi

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
echo "network:LiDAR_RCNN, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} images/sec, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log