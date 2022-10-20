#!/bin/bash
set -e
CUR_DIR=$(cd $(dirname $0);pwd)

function usage
{
    echo "Usage:"
    echo "-------------------------------------------------------------"
    echo "|  $0 [0-18] precision-device-[options...]"
    echo "|  required param1: 0)ERFNet,         1)ICNet,        2)SegFormer,          3)Twins,"
    echo "|                   4)FCN,            5)PSPNet,       6)BiSeNet,            7)FCN_R50,"       
    echo "|                   8)DeeplabV3_R50,  9)GCNet,       10)OCRNet,            11)UPERNet,"
    echo "|                  12)HRNet,         13)ISANet,      14)APCNet,            15)CCNet,"   
    echo "|                  16)CGNet,         17)DANet,       18)DMNet,             19)DNLNet,"   
    echo "|                  20)Stdc,          21)Beit,        22)Convnext,          23)Dpt"   
    echo "|                  24)EMANet,        25)ENCNet,      26)Fastfcn,           27)Fastfscnn"   
    echo "|                  28)Knet,          29)Mae,         30)MobileNet,         31)NonlocalNet"   
    echo "|                  32)Point_rend,    33)PSANet,      34)RESNest,           35)Sem_fpn"
    echo "|                  36)Setr,          37)Swin,        38)Vit"   
    echo "|  required param2: precision: fp32, O0, O1, O2, O3, amp "
    echo "|                   device: mlu, gpu            "
    echo "|                   option1: ddp                "
    echo "|  required param3: device nums                 "
    echo "|  eg. ./test_segmentation.sh 0 fp32-mlu"
    echo "|      which means running ERFNet on single MLU card with fp32 precision."
    echo "|  eg. export MLU_VISIBLE_DEVICES=0,1,2,3 && ./test_segmentation.sh 1 fp32-mlu-ddp"
    echo "|      which means running ICNet on 4 MLU cards with fp32 precision."
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
if [[ $net_index =~ ^[0-9]+ &&\
 ($configs =~ ^(fp32|O0|O1|O2|O3|amp)-(mlu|gpu)(-ddp)?(-ci.*)?$ || $configs =~ ^ci[_a-z]*$) ]]; then
    echo "Paramaters Exact."
else
    echo "[ERROR] Unknow Parameter : " $net_index $configs
    usage
    exit 1
fi

# get location of net
net_list=(  ERFNet          ICNet       SegFormer       Twins 
            FCN             PSPNet      BiSeNet         FCN_R50
            DeeplabV3_R50   GCNet       OCRNet          UPERNet
            HRNet           ISANet      APCNet          CCNet
            CGNet           DANet       DMNet           DNLNet
            Stdc            Beit        Convnext        Dpt
            EMANet          ENCNet      Fastfcn         Fastfscnn
            Knet            Mae         MobileNet       NonlocalNet
            Point_rend      PSANet      RESNest         Sem_fpn
            Setr            Swin        Vit
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

echo $DOCKERFILE
echo $RUN_MODEL_FILE

run_cmd="bash ../../launch.sh ${net_name} ${BATCH_SIZE} ${devs_all} ${precision} false ignore_check"

echo $run_cmd
eval $run_cmd
unset MLU_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES