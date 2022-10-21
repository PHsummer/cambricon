# A demo for PyTorch ResNet50v1.5 single card training, batch_size is fixed at 112 in source code.

# Choose GPU type
if [[ $gpu =~ "A100" ]]
then
    DGXSYSTEM="DGXA100"
elif [[ $gpu =~ "V100" ]]
then
    DGXSYSTEM="DGX1V"   
fi 

# For TF32 case
if [[ $gpu =~ "A100" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi

# Set batch size
if [[ $gpu =~ "A100" ]]
then
    if [[ $precision == "FP32" || $precision == "TF32" ]] && [[ ${batch_size} != "256" ]]
    then
        docker exec -it "${CONT_NAME}" bash -c "sed -i "155s/256/${batch_size}/g" ./configs.yml" 
    elif [[ $precision == "AMP" ]] && [[ ${batch_size} != "256" ]]
    then
        docker exec -it "${CONT_NAME}" bash -c "sed -i "150s/256/${batch_size}/g" ./configs.yml"
    fi
elif [[ $gpu =~ "V100" ]]
then
    if [[ $precision == "FP32" ]] && [[ ${batch_size} != "112" ]]
    then
        docker exec -it "${CONT_NAME}" bash -c "sed -i "131s/112/${batch_size}/g" ./configs.yml"
    elif [[ $precision == "AMP" ]] && [[ ${batch_size} != "256" ]]
    then 
        docker exec -it "${CONT_NAME}" bash -c "sed -i "127s/112/${batch_size}/g" ./configs.yml"
    fi
fi

# Run the model
if [[ $device_count == 1 ]] || [[ $device_count > 1 && $gpu =~ "V100" ]]
then
    docker exec -it "${CONT_NAME}" bash -c "python ./launch.py --model ${model_name} --precision ${precision} --mode benchmark_training --platform ${DGXSYSTEM} /data/pytorch/datasets/imagenet_training --raport-file benchmark.json --epochs 1 --prof 100" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    docker exec -it "${CONT_NAME}" bash -c "python ./multiproc.py --nproc_per_node ${device_count} ./launch.py --model ${model_name} --precision ${precision} --mode benchmark_training --platform ${DGXSYSTEM} /data/pytorch/datasets/imagenet_training --raport-file benchmark.json --epochs 1 --prof 100" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
throughput=$(grep  "train.total_ips"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -1 | cut -d " " -f 16)
throughput_all=$(echo "$throughput*$device_count"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
if [[ $gpu == "A100-SXM4-80GB" ]]
then
    if [[ $device_count == 1 ]]
    then
        if [[ $precision == "TF32" ]]
        then
            Github_Perf_Data="938(BS=112;${precision})"
        elif [[ $precision == "AMP" ]]
        then
            Github_Perf_Data="2470(BS=112;${precision})"
        fi
    elif [[ $device_count == 8 ]]
    then
        if [[ $precision == "TF32" ]]
        then
            Github_Perf_Data="7248(BS=112;${precision})"
        elif [[ $precision == "AMP" ]]
        then
            Github_Perf_Data="16621(BS=112;${precision})"
        fi
    fi
elif [[ $gpu == "V100-SXM2-16GB" ]]
then
    if [[ $device_count == 1 ]]
    then
        if [[ $precision == "FP32" ]]
        then
            Github_Perf_Data="367(BS=112;${precision})"
        elif [[ $precision == "AMP" ]]
        then
            Github_Perf_Data="1200(BS=112;${precision})"
        fi
    elif [[ $device_count == 8 ]]
    then
        if [[ $precision == "FP32" ]]
        then
            Github_Perf_Data="2855(BS=112;${precision})"
        elif [[ $precision == "AMP" ]]
        then
            Github_Perf_Data="8322(BS=112;${precision})"
        fi
    fi
fi
       
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
else
  dpf_mode="DDP"
fi

# Write benchmark log into a file
echo "network:ResNet-50v1.5, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} img/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
