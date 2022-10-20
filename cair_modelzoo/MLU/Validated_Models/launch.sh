set -e
CUR_DIR=$(cd `dirname $0`; pwd)
            
# Vars 
: "${WORK_DIR:=$CUR_DIR}"
: "${LOG_DIR:=$WORK_DIR/logs}"
: "${RESULT_DIR:=$WORK_DIR/results}"
: "${CNDB_DIR:=$WORK_DIR/tools/cndb}" 
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"


# Model Parameters
model_name=${1:-"resnet50"}
batch_size=${2:-"112"}
device_count=${3:-"1"}
precision=${4:-"FP32"}
database=${5:-"false"}
ignore_check=${6:-"false"}


if (( $devs_all<1 )) || (( $device_count<1 ))  || (( $device_count>$devs_all )) ;then
  echo -e "\033[31m device_count setting error! \033[0m"
  exit 1
fi

# Setup directories
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

# Build test docker images
CONT="yellow.hub.cambricon.com/cair_modelzoo/${dev_type}_benchmark:${model_name}-ctr${ctr_version}-pytorch-py3"
CONT_REPO=$(echo ${CONT} | awk -F: '{print $1}') 
CONT_TAG=$(echo ${CONT} | awk -F: '{print $2}') 
CONT_FIND=$(docker images | grep ${CONT_REPO} | grep " ${CONT_TAG}" )|| true
if [ ! -n "$CONT_FIND" ]; then
  if [[ ${dev_type} == "mlu" ]];then 
    docker build --network=host --build-arg FROM_IMAGE_NAME=${BASE_IMAGE_NAME} --build-arg MODEL_NAME=${NET_FOLDER} --build-arg DEVICE_TYPE="_${dev_type}" -t ${CONT} -f ${DOCKERFILE} .
  else
    docker build --network=host --build-arg FROM_IMAGE_NAME=${BASE_IMAGE_NAME} --build-arg MODEL_NAME=${NET_FOLDER} -t ${CONT} -f ${DOCKERFILE} .
  fi
else
  echo -e "\033[33m Docker image is found locally! \033[0m"
fi
# Devices info
if [[ ${dev_type} == "gpu" ]]; then 
  dev_name=$(nvidia-smi -L | head -n 1 | cut -d " " -f 4)
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv | tail -n 1)
elif [[ ${dev_type} == "mlu"  ]]; then
  dev_name=$(cnmon info -c 0 | grep "Product Name " | cut -d ":" -f 2)
  driver=$(cnmon info -c 0 | grep "Driver" | cut -d ":" -f 2)
fi

# PCIe inofs
if [[ ${dev_type} == "gpu" ]]; then
  pcie_speed=$(nvidia-smi -q -i 0 | grep "Current" | cut -d':' -f 2 | cut -d " " -f 2 | tail -n -4 | head -n 1 | cut -d "x" -f 1)
elif [[ ${dev_type} == "mlu"  ]]; then
  pcie_speed=$(cnmon info -c 0 | grep "Current Speed" | cut -d':' -f 2 | cut -d " " -f 2)
fi

case ${pcie_speed} in
  "8")
    pcie="gen3"
    ;;
  "16")
    pcie="gen4"
    ;;
  "32")
    pcie="gen5"
    ;;
  *)
    pcie="N/A"
    ;;
esac

# Docker container name
readonly docker_image=${CONT:-"nvcr.io/SET_THIS_TO_CORRECT_CONTAINER_TAG"}
CONT_NAME="${dev_type}_validated_models_$model_name"

# Cleanup container
cleanup_docker() {
    docker container rm -f "${CONT_NAME}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

if [[ ${dev_type} == "gpu" ]]; then
  BENCHMARK_LOG="benchmark_${dev_type}_${pcie}_cuda${cuda_version}_pt${torch_ver}"
elif [[ ${dev_type} == "mlu"  ]]; then
  BENCHMARK_LOG="benchmark_${dev_type}_${pcie}_ctr${ctr_version}_pt${torch_ver}"
fi

# mapping INIT_CHECKPOINT
if [ -n "$INIT_CHECKPOINT" ]; then 
  INIT_CHECKPOINT_MAPPING="-v ${INIT_CHECKPOINT}:/model"
else
  INIT_CHECKPOINT_MAPPING=""
fi
# Start container
if [[ ${dev_type} == "gpu" ]]; then
  docker run --rm --detach --net=host --ipc=host --shm-size=64g --ulimit memlock=-1 \
  --ulimit stack=67108864 --gpus all -v ${DATASETS_DIR}:/data  \
  $INIT_CHECKPOINT_MAPPING --name "${CONT_NAME}" "${docker_image}" sleep infinity
elif [[ ${dev_type} == "mlu"  ]]; then
  docker run --rm --detach --net=host --ipc=host --shm-size=64g --ulimit memlock=-1 \
  --ulimit stack=67108864 --device /dev/cambricon_ctl -v ${DATASETS_DIR}:/data \
  $INIT_CHECKPOINT_MAPPING --privileged --name "${CONT_NAME}" "${docker_image}" sleep infinity
fi
#make sure container has time to finish initialization
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10
docker exec -it "${CONT_NAME}" true


# Check GPU performance environment
if [[ ${ignore_check} == "ignore_check" ]]
then
    echo -e "\033[31m Warning : You are igoring performance environment checking steps, which may cause training performance degradation!\033[0m" 
    if [[ ${dev_type} == "gpu" ]]; then
      bash ${WORK_DIR}/../../GPU/tools/check_gpu_perf.sh ignore_check
    elif [[ ${dev_type} == "mlu"  ]]; then
      bash ${WORK_DIR}/tools/check_mlu_perf.sh ignore_check
    fi
else
    bash ${WORK_DIR}/tools/check_mlu_perf.sh
    echo -e "\033[32m Check GPU performance environment done! \033[0m"
fi

# Run experiments: store benchmark messages to log
source ${RUN_MODEL_FILE}
echo -e "\033[32m Run experiments done! \033[0m"

# Convert benchmark_log to cndb format
# framework_image=${BASE_IMAGE_NAME}
# python ${WORK_DIR}/tools/dump.py \
#   -i ${RESULT_DIR}/${BENCHMARK_LOG} \
#   -o ${RESULT_DIR} \
#   --framework ${framework_image}  \
#   --code_link ${CODE_LINK}
# echo -e "\033[32m Convert benchmark_log done! \033[0m"

# Upload results to superset
# if [[ ${database} != false ]]
# then
#     if [[ ${database} == "mlu-validated" ]]
#     then
# 	      db_file="mlu-validated-models.yaml"
#     elif [[ ${database} == "mlu-demo" ]]
#     then
#         db_file="mlu-validated-demo.yaml"	    
#     fi
#     for file in $(ls ${RESULT_DIR}/*yaml); do
#       cndb_submit \
#         --db_file ${CNDB_DIR}/demo/config/${db_file}  \
#         --load_file ${file} \
#         --log-level INFO \
#         --db_store_type save
#     done
#     echo -e "\033[32m Upload results done! You can check your training results on http://dataview.cambricon.com now! \033[0m"
# else
#     echo -e "\033[32m You choose to not upload the results to database! \033[0m"
# fi

# Move performance yaml to logs file and delete other output files"
# save_dir=$(echo ${DOCKERFILE%%Dockerfile})
# mkdir -p "${save_dir}logs"
# mv ${RESULT_DIR}/*yaml ${save_dir}/logs/
#rm ${RESULT_DIR} -fr

echo -e "\033[32m All Done! \033[0m"
