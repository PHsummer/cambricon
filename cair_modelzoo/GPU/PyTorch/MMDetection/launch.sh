set -e
CUR_DIR=$(cd `dirname $0`; pwd)
            
# Vars 
: "${CONT:="mmdetection_docker"}"         
: "${DOCKERFILE:="./Dockerfile"}"
: "${WORK_DIR:=$CUR_DIR}"
: "${DATASETS_DIR:=$WORK_DIR/data}"
: "${LOG_DIR:=$WORK_DIR/logs}"
: "${TEST_DIR:=$WORK_DIR/tests}"
: "${CONFIG_DIR:=$CUR_DIR/checkpoints}"


# Build test docker images
CONT_REPO=$(echo ${CONT} | awk -F: '{print $1}') 
CONT_TAG=$(echo ${CONT} | awk -F: '{print $2}') 
CONT_FIND=$(docker images | grep ${CONT_REPO} | grep ${CONT_TAG}) || true
if [ ! -n "$CONT_FIND" ]
then
  docker build -t ${CONT} -f ${DOCKERFILE} .
else
  echo "Docker image is found locally!"
fi

# start container

num=`docker ps -a | grep "${CONT}" | wc -l`

if [ '0' -eq $num ]
then
  docker run --gpus all -id --name=${CONT} -v ${DATASETS_DIR}:/mmdetection/tests/data -v ${LOG_DIR}:/mmdetection/logs -v ${CONFIG_DIR}:/mmdetection/checkpoints -v ${TEST_DIR}:/mmdetection/tests ${CONT}  
echo "container is creating..."

else
  docker restart "${CONT}"
fi

echo "Run container done!"


# test mmdetection
(cd checkpoints;wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth)
docker exec -it "${CONT}" /bin/bash -c "python tests/test_mmdetection.py"

echo "All Done!"
