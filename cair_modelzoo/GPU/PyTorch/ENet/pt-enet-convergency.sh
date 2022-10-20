set -e
CUR_DIR=$(cd `dirname $0`; pwd)

# Setup directories
: "${WORK_DIR:=$CUR_DIR}"
: "${LOG_DIR:=$WORK_DIR/logs}"
mkdir -p "${WORK_DIR}"
mkdir -p "${LOG_DIR}"

model_name_convergency=${1:-"enet"}

# Vars 
: "${DATASETS_DIR:="/algo/modelzoo/datasets/"}"   
: "${CONT:="yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:enet-21.06-pytorch-py3"}"
: "${DOCKERFILE:="./PyTorch/ENet/Dockerfile"}"
: "${DATESTAMP:=$(date +'%y%m%d%H%M%S%N')}"

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

readonly docker_image=${CONT:-"nvcr.io/SET_THIS_TO_CORRECT_CONTAINER_TAG"}
CONT_NAME="gpu_golden_bench_$model_name_convergency"

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

# Run experiments
docker exec -it "${CONT_NAME}" bash -c "git apply --ignore-space-change --ignore-whitespace --reject modify.patch && python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 main.py -m train --deterministic --batch-size 2 --epochs 300 --save-dir ./checkpoints --name enet --dataset cityscapes --dataset-dir /data/CityScapes" 2>&1 | tee ${LOG_DIR}/enet-convergency_${DATESTAMP}.log 
echo -e "\033[32m Run experiments done! \033[0m"