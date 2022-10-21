# start docker container
export MY_CONTAINER="test_pytorch_bert-large-8cards"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [ 0 -ne $num ];then 
  docker container rm -f "${MY_CONTAINER}" || true
fi
docker run -itd --rm \
        --shm-size '64gb' \
        --device /dev/cambricon_dev0 \
        --device /dev/cambricon_dev1 \
        --device /dev/cambricon_ctl \
        --name $MY_CONTAINER \
        -v `pwd`:/mnt \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        -v /data:/data \
        --privileged \
        yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:bert-large-catch1.1.2-1-x86_64-ubuntu18.04-run \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
docker exec -it $MY_CONTAINER bash -c "source /torch/venv3/pytorch/bin/activate && bash x8-8cards-from-scratch.sh"
