# A demo for PyTorch pix2pix single card training, batch_size is fixed at 112 in source code.

# For TF32 case
if [[ $gpu =~ "A100" || $gpu =~ "3090" ]] && [[ $precision == "FP32" ]]
then
    precision="TF32"
fi

#Set GPU ids
max_gpu_index=$(echo "${device_count}-1"|bc)
gpu_list=$(seq 0 ${max_gpu_index} | xargs -n 16 echo | tr ' ' ',')

# Run the model
if [[ $device_count == 1 ]] 
then
    docker exec -it "${CONT_NAME}" bash -c "python train.py --dataroot ./datasets/facades --name facades_pix2pix --model ${model_name} --netG resnet_9blocks --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --batch_size ${batch_size}  --phase train --n_epochs 1 --n_epochs_decay 0 --print_freq 1 --gpu_ids ${gpu_list}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log
else
    docker exec -it "${CONT_NAME}" bash -c "python train.py --dataroot ./datasets/facades --name facades_pix2pix --model ${model_name} --netG resnet_9blocks --direction BtoA --lambda_L1 100 --dataset_mode aligned --norm batch --pool_size 0 --batch_size ${batch_size}  --phase train --n_epochs 1 --n_epochs_decay 0 --print_freq 1 --gpu_ids ${gpu_list}" 2>&1 | tee ${LOG_DIR}/${model_name}_${DATESTAMP}.log    
fi

# Get performance results
throughput=$(grep "epoch: 1" ${LOG_DIR}/${model_name}_${DATESTAMP}.log | sed -n "3,7p" | awk '{print$6}' | sed 's/\,//g' | awk '{sum+=$1}END{print 1/(sum/5)}')

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"
       
# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
else
  dpf_mode="DP"
fi

# Write benchmark log into a file
echo "network:pix2pix-resnet_9blocks, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, throughput:${throughput} img/s, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
