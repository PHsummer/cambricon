# start docker container
export MY_CONTAINER="test_tf_bert-base-1X8"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
if [ 0 -eq $num ];then 
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
        yellow.hub.cambricon.com/tensorflow/tensorflow:v1.10.1-x86_64-ubuntu1804-py3 \
        /bin/bash
echo -e "\033[33m Training container is creating... \033[0m"
sleep 10

docker exec -it "${MY_CONTAINER}" true
docker exec -it $MY_CONTAINER bash -c "cd tensorflow_benchmark/benchmarks/cn_benchmarks/TensorFlow/NaturalLanguageProcessing/BERT/google_bert &&  bash launch.sh -m train_and_eval -n google_bert -b 28 -a True -t mlu_model -h True -p "np:2,learning_rate:1.05e-5""

