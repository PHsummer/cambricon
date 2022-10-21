set -e
CUR_DIR=$(cd `dirname $0`; pwd)
            
# Vars 
: "${CONT:="yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:resnet50v1.5-22.01-pytorch-py3"}"
: "${DATASETS:="ImageNet2012"}"    
: "${DATASETS_DIR:="/data"}"         
: "${CODE_LINK:="https://github.com/NVIDIA/DeepLearningExamples/tree/3a8068b651c8ae919281b638166b3ecfa07d22f5/PyTorch/Classification/ConvNets"}"
: "${RUN_MODEL_FILE:="./PyTorch/ResNet50v1.5/pt-resnet-50v1.5.sh"}"
: "${DOCKERFILE:="./PyTorch/ResNet50v1.5/Dockerfile"}"
: "${WORK_DIR:=$CUR_DIR}"
: "${LOG_DIR:=$WORK_DIR/logs}"
: "${RESULT_DIR:=$WORK_DIR/results}"
: "${CNDB_DIR:?CNDB_DIR not set}" 
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"
: "${CONFIG:="./PyTorch/ResNet50v1.5/configs"}"

usage() {
  echo "Usage:"
  echo "CONT=<docker image name> DATASETS=<datasets name> DATASETS_DIR=<datasets dir> CNDB_DIR=<cndb path> CODE_LINK=<code link> RUN_MODEL_FILE=<model shell script> DOCKERFILE=<Dockerfile> ./$(basename $0) <model_name> <batch_size> <device_count> <precision>"
  exit -1
}

# Model Parameters
model_name=${1:-"resnet50"}
batch_size=${2:-"112"}
device_count=${3:-"1"}
precision=${4:-"FP32"}
database=${5:-"false"}
ignore_check=${6:-"false"}

# Parameter check
gpus_all=$(nvidia-smi -L | wc -l)
if (( $gpus_all<1 )) || (( $device_count<1 )) || (( $device_count>$gpus_all ))
then
  echo -e "\033[31m device_count setting error! \033[0m"
  exit 1
fi

# Setup directories
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"
mkdir -p "${RESULT_DIR}"

# Build test docker images
CONT_REPO=$(echo ${CONT} | awk -F: '{print $1}') 
CONT_TAG=$(echo ${CONT} | awk -F: '{print $2}') 
CONT_FIND=$(docker images | grep ${CONT_REPO} | grep ${CONT_TAG}) || true
if [ ! -n "$CONT_FIND" ]
then
  docker build -t ${CONT} -f ${DOCKERFILE} .
else
  echo -e "\033[33m Docker image is found locally! \033[0m"
fi

# GPU infos
gpu=$(nvidia-smi -L | head -n 1 | cut -d " " -f 4)
driver=$(nvidia-smi --query-gpu=driver_version --format=csv | tail -n 1)

readonly docker_image=${CONT:-"nvcr.io/SET_THIS_TO_CORRECT_CONTAINER_TAG"}
CONT_NAME="gpu_golden_bench_$model_name"

# Cleanup container
cleanup_docker() {
    docker container rm -f "${CONT_NAME}" || true
}
cleanup_docker
trap 'set -eux; cleanup_docker' EXIT

# Start container
docker run --rm --detach --gpus all --net=host --ipc=host --shm-size=64g --ulimit memlock=-1 \
--ulimit stack=67108864 -v ${DATASETS_DIR}:/data  \
--name "${CONT_NAME}" "${docker_image}" sleep infinity
#make sure container has time to finish initialization
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10
docker exec -it "${CONT_NAME}" true

# Check GPU performance environment
if [[ ${ignore_check} == "ignore_check" ]]
then
    echo -e "\033[31m Warning : You are igoring performance environment checking steps, which may cause training performance degradation!\033[0m" 
    bash tools/check_gpu_perf.sh ignore_check
else
    bash tools/check_gpu_perf.sh
    echo -e "\033[32m Check GPU performance environment done! \033[0m"
fi

# Run experiments: store benchmark messages to log
source ${RUN_MODEL_FILE}
echo -e "\033[32m Run experiments done! \033[0m"

# Convert benchmark_log to cndb format
python dump_for_cndb.py \
  -i ${RESULT_DIR}/gpu_benchmark_log \
  -o ${RESULT_DIR} \
  --ngc_name ${CONT} \
  --code_link ${CODE_LINK}
echo -e "\033[32m Convert benchmark_log done! \033[0m"

# Upload results to superset
if [[ ${database} != false ]]
then
    if [[ ${database} == "gpu-performance" ]]
    then
	db_file="gpu-golden-bench.yaml"
    elif [[ ${database} == "gpu-convergency" ]]
    then	    
	db_file="gpu-golden-bench-convergency.yaml"
    elif [[ ${database} == "gpu-demo" ]]
    then
        db_file="demo-db.yaml"	    
    fi
    for file in $(ls ${RESULT_DIR}/*yaml); do
      cndb_submit \
        --db_file ${CNDB_DIR}/demo/config/${db_file} \
        --load_file ${file} \
        --log-level INFO \
        --db_store_type save
    done
    echo -e "\033[32m Upload results done! You can check your training results on http://dataview.cambricon.com/superset/dashboard now! \033[0m"
else
    echo -e "\033[32m You choose to not upload the results to database! \033[0m"
fi

# Move performance yaml to logs file and delete other output files"
mv ${RESULT_DIR}/*yaml ${LOG_DIR}/
rm ${RESULT_DIR} -fr

echo -e "\033[32m All Done! \033[0m"
