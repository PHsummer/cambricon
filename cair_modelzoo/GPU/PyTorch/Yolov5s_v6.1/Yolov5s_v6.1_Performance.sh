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

#Set GPU ids
max_gpu_index=$(($device_count -1))
gpu_list=$(seq 0 $max_gpu_index | xargs -n 16 echo | tr ' ' ',')

# Run the model
if [[ $device_count == 1 ]]
then
    docker exec -it "${CONT_NAME}"  bash -c "python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size ${batch_size} --epochs 2 --device 0 " 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
    iter_total=705
else
    docker exec -it "${CONT_NAME}"  bash -c "python -m torch.distributed.run --nproc_per_node ${device_count} train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size ${batch_size} --epochs 2 --device $gpu_list" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
    iter_total=89
fi

# Get performance results
cut_idx=$(($iter_total*2+3))
iter_fps=$(grep " 1/1 " ${LOG_DIR}/${model_name}_${DATESTAMP}.log | cut -d "<" -f $cut_idx | cut -d " " -f 3 | cut -d "/" -f 1)
if [[ ${iter_fps:4:5} == s ]]
then 
    throughput_all=$(echo "${batch_size} / ${iter_fps:0:4}"|bc)
else 
    throughput_all=$(echo "${batch_size} * ${iter_fps:0:4}"|bc)
fi
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
echo "network:Yolov5s_v6.1, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} fps, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
