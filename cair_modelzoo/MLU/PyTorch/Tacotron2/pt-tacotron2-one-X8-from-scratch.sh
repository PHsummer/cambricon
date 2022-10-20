# start docker container
export MY_CONTAINER="test_pytorch_tacotron2-1X8_accuracy"
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
        -v `pwd`:/mnt \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        -v /data:/data \
        --privileged \
        yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:tacotron2-catch1.2.1-torch1.9-x86_64-ubuntu18.04 \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
#如果报显存不够，可以先删除docker容器，再删除镜像，修改“scratch_tacotron2_2mlu_AMP.sh”中batch size为96，然后再按照README中从头操作。
docker exec -it $MY_CONTAINER bash -c "source /torch/venv3/pytorch/bin/activate && cd Tacotron2andWaveGlow/cambricon && bash scratch_tacotron2_2mlu_AMP.sh"

