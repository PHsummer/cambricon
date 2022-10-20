export MY_CONTAINER="zsl_pt19_v100"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -it \
         --shm-size '64gb' \
         --gpus all \
         --name $MY_CONTAINER \
        -v `pwd`:/mnt \
        -v /data1:/data1 \
        -v /home/zhangshuanglin:/home/zhangshuanglin \
        --privileged=true \
        -u root \
        nvcr.io/nvidia/pytorch:20.07-py3 \
        /bin/bash
else
  docker start $MY_CONTAINER
  docker exec -ti $MY_CONTAINER /bin/bash
fi