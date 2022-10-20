export MY_CONTAINER="cz_pt16_1"
num=`docker ps -a|grep "$MY_CONTAINER"|wc -l`
echo $num
echo $MY_CONTAINER
if [ 0 -eq $num ];then
docker run -it \
         --shm-size '64gb' \
         --device /dev/cambricon_dev0 \
         --device /dev/cambricon_dev1 \
         --device /dev/cambricon_dev2 \
         --device /dev/cambricon_dev3 \
         --device /dev/cambricon_dev4 \
         --device /dev/cambricon_dev5 \
         --device /dev/cambricon_dev6 \
         --device /dev/cambricon_dev7 \
         --device /dev/cambricon_dev8 \
         --device /dev/cambricon_dev9 \
         --device /dev/cambricon_dev10 \
         --device /dev/cambricon_dev11 \
         --device /dev/cambricon_dev12 \
         --device /dev/cambricon_dev13 \
         --device /dev/cambricon_dev14 \
         --device /dev/cambricon_dev15 \
         --name $MY_CONTAINER \
        -v `pwd`:/mnt \
        -v /data:/data \
        -v /usr/bin/cnmon:/usr/bin/cnmon \
        -v /home/cuizhe:/workspace \
        --privileged=true \
        yellow.hub.cambricon.com/pytorch/pytorch:v1.6.0-torch1.6-ubuntu20.04  \
        /bin/bash
else
  docker start $MY_CONTAINER
  docker exec -ti $MY_CONTAINER /bin/bash
fi
