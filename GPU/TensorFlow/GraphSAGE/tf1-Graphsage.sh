#Run the model
if [[ $device_count == 1 && $gpu =~ "V100" ]]
then
    if [[ $model_name == "unsupervised_graphsage_mean" ]]
    then
        docker exec -it "${CONT_NAME}" bash -c "python -m graphsage.unsupervised_train --train_prefix data/ppi/ppi --model graphsage_mean --max_total_steps 10000 --validate_iter 10 --epochs 10 --batch_size=${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
    elif [[ $model_name == "supervised_graphsage_mean" ]]
    then
        docker exec -it "${CONT_NAME}" bash -c "python -m graphsage.supervised_train --train_prefix data/ppi/ppi --model graphsage_mean --sigmoid --epochs 100 --batch_size=${batch_size}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
    fi
elif [[$device_count > 1 && $gpu =~ "V100"]]
then
    echo "Multiproc does not exit."
    return 1
    break
# docker exec -it "${CONT_NAME}" bash -c "python ./multiproc.py --nproc_per_node ${device_count} ./launch.py --model ${model_name} --precision ${precision} --mode benchmark_training --platform ${DGXSYSTEM} /data/pytorch/datasets/imagenet_training --raport-file benchmark.json --epochs 1 --prof 100" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
throughput=$(grep  "rate_avg"  ${LOG_DIR}/${model_name}_${DATESTAMP}.log | tail -n -1 | awk '{print $18}')
throughput_all=$(echo "$throughput"|bc)

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"

# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
fi



# Write benchmark log into a file
if [[ $model_name == "unsupervised_graphsage_mean" ]]
then
    echo "network:GraphSage-unsupervised-mean, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} nodes/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
elif [[ $model_name == "supervised_graphsage_mean" ]]
then
    echo "network:GraphSage-supervised-mean, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput_all} nodes/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
fi
