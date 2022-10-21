# start docker container
export MY_CONTAINER="test_pytorch_dlrm-1X8_accuracy"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [ 0 -ne $num ];then 
  docker container rm -f "${MY_CONTAINER}" || true
fi
docker run -itd \
        --shm-size '64gb' \
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_dev1 \
        --device /dev/cambricon_ctl \
        --name $MY_CONTAINER \
        -e PYTORCH_TRAIN_DATASET=/data/pytorch/datasets \
        -e PYTORCH_TRAIN_CHECKPOINT=/data/pytorch/models/pytorch_weight/checkpoints \
        -v `pwd`:/mnt \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        -v /data:/data \
        --privileged \
        yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:dlrm-catch1.2.1-torch1.6-x86_64-ubuntu18.04 \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
docker exec -it $MY_CONTAINER bash -c "source /torch/venv3/pytorch/bin/activate && cd DLRM/cambricon && bash scratch_DLRM_2mlu.sh 0"
docker exec -it $MY_CONTAINER bash -c "source /torch/venv3/pytorch/bin/activate && cd DLRM/cambricon && bash scratch_DLRM_2mlu_test.sh 0"
