# Run experiments
docker exec -it "${CONT_NAME}" bash -c "python train.py --dataroot ./datasets/facades --name facades_pix2pix_resnet_9blocks --model ${model_name} --netG resnet_9blocks --direction BtoA" 2>&1 | tee ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log
docker exec -it "${CONT_NAME}" bash -c "python test.py --dataroot ./datasets/facades --name facades_pix2pix_resnet_9blocks --model ${model_name} --netG resnet_9blocks --direction BtoA" 2>&1 | tee ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}_test.log
echo -e "\033[32m Run experiments done! \033[0m"

# Get convergence results
G_GAN=$(grep "epoch: 200" ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log | awk '{sum+=$10} END {print "", sum/NR}')
G_L1=$(grep "epoch: 200" ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log | awk '{sum+=$12} END {print "", sum/NR}')
D_real=$(grep "epoch: 200" ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log | awk '{sum+=$14} END {print "", sum/NR}')
D_fake=$(grep "epoch: 200" ${LOG_DIR}/bert-crf-convergency_${DATESTAMP}.log | awk '{sum+=$16} END {print "", sum/NR}')
accuracy="G_GAN =${G_GAN}  G_L1 =${G_L1}  D_real =${D_real}  D_fake =${D_fake}"

# DPF mode
if [[ $device_count == 1 ]]
then
  dpf_mode="Single"
else
  dpf_mode="DP"	
fi

# NV Performance Data
NV_Web_Perf_Data="N/A"
Github_Perf_Data="N/A"

# Write benchmark log into a file
echo "network:${model_name}, batch size:${batch_size}, device count:${device_count}, dpf mode:${dpf_mode}, precision:${precision}, accuracy:$accuracy, device:${gpu}, dataset:${DATASETS}, nv_web_perf:${NV_Web_Perf_Data}, github_perf_data:${Github_Perf_Data}, driver:${driver}" >> ${RESULT_DIR}/gpu_benchmark_log
