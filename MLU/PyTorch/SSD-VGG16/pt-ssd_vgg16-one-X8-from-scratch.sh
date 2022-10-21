# start docker container
export MY_CONTAINER="test_pytorch_ssd-vgg16-1X8_accuracy"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [[ 0 -ne $num ]];then 
  echo -e "\033[33m Deleting the existing container... \033[0m"
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
        yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:ssd_vgg16-catch1.2.1-torch1.6-x86_64-ubuntu18.04 \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
# pytorch:v1.2.1-torch1.6-ubuntu18.04此处代码有问题，已经和guwei反馈过。其他版本按需修改。
docker exec -it $MY_CONTAINER bash -c "sed -i '219s/$/\n    if args.pyamp:/g' ssd_vgg16/train.py && sed -i '220s/$/\n        scaler = GradScaler()/g' ssd_vgg16/train.py"

docker exec -it $MY_CONTAINER bash -c "source /torch/venv3/pytorch/bin/activate && cd ssd_vgg16/cambricon && pip install tensorboardX && bash scratch_2mlu_amp.sh"

