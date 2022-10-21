# start docker container
export MY_CONTAINER="test_tf2_resnet50-1X8"
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
        tensorflow2-1.10.1-x86_64-ubuntu18.04:latest \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
docker exec -it $MY_CONTAINER bash -c "cd tensorflow_benchmark/benchmarks/cn_benchmarks/TensorFlow2/Classification/Resnet50_CMCC && pip install -r requirements.txt && bash launch.sh -m train_and_eval -n resnet50 -b 256 -t mlu_model -a True -h True -p "np:2,base_learning_rate:0.2""

