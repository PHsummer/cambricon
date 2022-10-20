#/bin/bash

export MY_CONTAINER="modelzoo_pytorch1.3.0"

num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
xhost +
docker run -e DISPLAY=unix$DISPLAY --device /dev/cambricon_ctl --network host --pid=host -v /sys/kernel/debug:/sys/kernel/debug -v /tmp/.X11-unix:/tmp/.X11-unix -it --privileged --name $MY_CONTAINER -v /data:/data -v $PWD/../:/home/resnet50v1.5 yellow.hub.cambricon.com/pytorch/pytorch:v1.3.1-torch1.6-ubuntu18.04  /bin/bash
else
docker start $MY_CONTAINER
#sudo docker attach $MY_CONTAINER
docker exec -ti $MY_CONTAINER /bin/bash
fi
