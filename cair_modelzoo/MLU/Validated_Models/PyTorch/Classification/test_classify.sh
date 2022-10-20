#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0-18] precision-device-[options...]"
    echo "|  required param1: 0)ecanet,         1)inception_resnetv2,  2)inceptionv4,          3)HRNet,"
    echo "|                   4)mobilenetv3,    5)ResNeXt,             6)ShuffleNetV2,         7)SqueezeNet,"
    echo "|                   8)WRN50V2,        9)Xception,            10)vovnet,              11)DPN68,"
    echo "|                   12)ShuffleNet_1x_g3,"
    echo "|  required param2: precision: fp32, O0, O1, O2, O3, amp "
    echo "|                   device: mlu, gpu            "
    echo "|                   option1: ddp                "
    echo "|  required param3: device nums                 "
    echo "|  eg. ./test_classify.sh 1 fp32-mlu"
    echo "|      which means running ecanet on single MLU card with fp32 precision."
    echo "|  eg. export MLU_VISIBLE_DEVICES=0,1,2,3 && ./test_classify.sh 1 fp32-mlu-ddp"
    echo "|      which means running ecanet on 4 MLU cards with fp32 precision."
    echo "-------------------------------------------------------------"
}

# Check numbers of argument
if [ $# -lt 2 ]; then
    echo "[ERROR] Not enough arguments."
    usage
    exit 1
fi

net_index=$1
configs=$2
precision=${configs%%-*}

# Paramaters check
if [[ $net_index =~ ^[0-9]+$ && $net_index -le 18 &&\
 ($configs =~ ^(fp32|O0|O1|O2|O3|amp)-(mlu|gpu)(-ddp)?(-ci.*)?$ || $configs =~ ^ci[_a-z]*$) ]]; then
    echo "Paramaters Exact."
else
    echo "[ERROR] Unknow Parameter : " $net_index $configs
    usage
    exit 1
fi

# get location of net
net_list=(  ECANet           Inception_ResNetv2    InceptionV4    HRNet 
            MobileNetV3      ResNeXt               ShuffleNetV2   SqueezeNet 
            WRN50V2          Xception              vovnet         DPN68 
            ShuffleNet_1x_g3
            )
net_name=${net_list[$net_index]}
net_location=${CUR_DIR}/${net_name}

checkstatus () {
    if (($?!=0)); then
        echo "work failed"
        exit -1
    fi
}

source ../../tools/params_config.sh
set_configs "${net_name}-${configs}"


if [[ $ddp == "True" ]]; then
    if [[ ${dev_type} == "gpu" ]]; then 
        devs_all=$(nvidia-smi -L | wc -l)
    elif [ ${dev_type} == "mlu"  ]; then
        devs=$(cnmon info | grep Card | tail -1 | cut -d " " -f 2)
        devs_all=$(echo "${devs}+1"|bc)
    fi
else
    devs_all=1
fi

export devs_all=${devs_all}
export DOCKERFILE="${CUR_DIR}/${NET_FOLDER}/Dockerfile"
export RUN_MODEL_FILE="${CUR_DIR}/${NET_FOLDER}/${net_name}_Performance.sh"

run_cmd="bash ../../launch.sh ${net_name} ${BATCH_SIZE} ${devs_all} ${precision} false ignore_check"

echo $run_cmd
eval $run_cmd
unset MLU_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES
