#!/bin/bash

CONFIG_DIR=$(cd $(dirname $0);pwd)

############
# Nets info
############
ECANet_base_params () {
    BATCH_SIZE=96
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/BangguWu/ECANet/tree/b332f6b3e6e2afe8a3287dc8ee8440a0fbec74c4"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

Inception_ResNetv2_base_params () {
    BATCH_SIZE=58
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

InceptionV4_base_params () {
    BATCH_SIZE=96
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

MobileNetV2_base_params () {
    BATCH_SIZE=300
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/d-li14/mobilenetv2.pytorch/tree/1733532bd43743442077326e1efc556d7cfd025d"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

MobileNetV3_base_params () {
    BATCH_SIZE=64
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

DPN68_base_params () {
    BATCH_SIZE=104
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

ERFNet_base_params () {
    BATCH_SIZE=11
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

ICNet_base_params () {
    BATCH_SIZE=24
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

SegFormer_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}


Vit_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Setr_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Swin_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Beit_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Twins_base_params () {
    BATCH_SIZE=6
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

GOTURN_base_params () {
    BATCH_SIZE=5700
    DATASETS='ILSVRC2014_DET & Alov300++'
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    CODE_LINK="https://github.com/amoudgl/pygoturn/tree/1785ca7106fc0aa9e0fd03b440f34454e8a10e78"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

MDNet_base_params () {
    BATCH_SIZE=1024
    DATASETS="ILSVRC2015"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    CODE_LINK="https://github.com/hyeonseobnam/py-MDNet/tree/680fa4d58c427b8b647fa59dd9cd61a9fb7061f6"
    
    PYTORCH_VER="19"
    benchmark_mode="True"
    }

MobileNetV2_SSDlite_base_params () {
    BATCH_SIZE=64
    DATASETS="COCO2017"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    CODE_LINK="https://github.com/open-mmlab/mmdetection/tree/ca11860f4f3c3ca2ce8340e2686eeaec05b29111"

    PYTORCH_VER="16"
    INTEGRATE="mmdetection"
    benchmark_mode="True"
}

Siamese_base_params () {
    BATCH_SIZE=896
    DATASETS="omniglot"
    DATASETS_DIR="/data/datasets/"
    CODE_LINK="https://github.com/fangpin/siamese-pytorch/tree/5543f1e844964b116dc9d347a5eb164c6a7afe6d"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

ST_GCN_base_params () {
    BATCH_SIZE=160
    DATASETS="Kinetics"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    CODE_LINK="https://github.com/fendou201398/st-gcn/tree/91e4046fe2274ac74d6220998996cdcd955ba715"

    PYTORCH_VER="19"
    benchmark_mode="True"
}


KG_Bert_base_params () {
    BATCH_SIZE_GPU=600
    BATCH_SIZE=600
    DATASETS="WN11"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    INIT_CHECKPOINT="/projs/platform/public/models"
    INIT_CHECKPOINT_GPU="/projs/platform/public/models"
    CODE_LINK="https://github.com/yao8839836/kg-bert/commit/0b8f625108efcb9b43603b6840f2548ff3d3c973"

    PYTORCH_VER="19"
    benchmark_mode="True"
}

ResNeXt_base_params () {
    BATCH_SIZE=170
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/pytorch/examples/tree/2bf23f105237e03ee2501f29670fb6a9ca915096"

    INTEGRATE='torchvision'
    PYTORCH_VER="19"
    benchmark_mode="True"
}

ShuffleNetV2_base_params () {
    BATCH_SIZE=480
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/pytorch/examples/tree/2bf23f105237e03ee2501f29670fb6a9ca915096"

    INTEGRATE='torchvision'
    PYTORCH_VER="19"
    benchmark_mode="True"
}

SqueezeNet_base_params () {
    BATCH_SIZE=450
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/pytorch/examples/tree/2bf23f105237e03ee2501f29670fb6a9ca915096"
    
    INTEGRATE='torchvision'
    PYTORCH_VER="19"
    benchmark_mode="True"
}

WRN50V2_base_params () {
    BATCH_SIZE=170
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/pytorch/examples/tree/2bf23f105237e03ee2501f29670fb6a9ca915096"

    INTEGRATE='torchvision'
    PYTORCH_VER="19"
    benchmark_mode="True"
}

Xception_base_params () {
    BATCH_SIZE=64
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

vovnet_base_params () {
    BATCH_SIZE=174
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"

    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

CenterNet_base_params () {
    BATCH_SIZE=38
    DATASETS="COCO2017"
    DATASETS_DIR="/data/pytorch/datasets/COCO2017"
    DATASETS_GPU="/data/datasets-common/COCO2017"
    CODE_LINK="https://github.com/xingyizhou/CenterNet/tree/2b7692c377c6686fb35e473dac2de6105eed62c6"
 
    PYTORCH_VER="19"
    benchmark_mode="True"
}

FaceBox_base_params () {
    BATCH_SIZE=160
    DATASETS="WIDER_FACE"
    DATASETS_DIR="/data/datasets"
    DATASETS_GPU="/data/datasets-common"
    CODE_LINK="https://github.com/zisianw/FaceBoxes.PyTorch/tree/9bc5811fe8c409a50c9f23c6a770674d609a2c3a"
 
    PYTORCH_VER="19"
    benchmark_mode="True"
}

HRNet_base_params () {
    BATCH_SIZE=96
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets"
    CODE_LINK="https://github.com/rwightman/pytorch-image-model/tree/e4360e6125bb0bb4279785810c8eb33b40af3ebd"
 
    PYTORCH_VER="19"
    INTEGRATE='timm'
    benchmark_mode="True"
}

Bert_base_params () {
    BATCH_SIZE_GPU=16
    BATCH_SIZE=16
    DATASETS="squad"
    DATASETS_DIR="/data/pytorch/datasets/BERT/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    INIT_CHECKPOINT="/projs/platform/public/models"
    INIT_CHECKPOINT_GPU="/projs/platform/public/models"
    CODE_LINK="https://github.com/huggingface/transformers/commit/cc5c061e346365252458126abb699b87cda5dcc0"

    PYTORCH_VER="16"
    benchmark_mode="True"
}

MacBert_base_params () {
    BATCH_SIZE_GPU=21
    BATCH_SIZE=21
    DATASETS="DRCD"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/data/datasets-common/"
    INIT_CHECKPOINT="/projs/platform/public/models"
    INIT_CHECKPOINT_GPU="/projs/platform/public/models"
    CODE_LINK="https://github.com/huggingface/transformers/commit/cc5c061e346365252458126abb699b87cda5dcc0"

    PYTORCH_VER="16"
    benchmark_mode="True"
}

RoBERTa_base_params () {
    BATCH_SIZE_GPU=33
    BATCH_SIZE=33
    DATASETS="squad"
    DATASETS_DIR="/data/pytorch/datasets/BERT/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    INIT_CHECKPOINT="/projs/platform/public/models"
    INIT_CHECKPOINT_GPU="/projs/platform/public/models"
    CODE_LINK="https://github.com/huggingface/transformers/commit/cc5c061e346365252458126abb699b87cda5dcc0"

    PYTORCH_VER="16"
    benchmark_mode="True"
}

FCN_base_params () {
    BATCH_SIZE=28
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}


FCN_base_params () {
    BATCH_SIZE=28
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

PSANet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
RESNest_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
Sem_fpn_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
PSPNet_base_params () {
    BATCH_SIZE=7
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

BiSeNet_base_params () {
    BATCH_SIZE=5
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

FCN_R50_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

DeeplabV3_R50_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

GCNet_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}


OCRNet_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
APCNet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
CCNet_base_params () {
    BATCH_SIZE=2
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
CGNet_base_params () {
    BATCH_SIZE=16
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
DANet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
DMNet_base_params () {
    BATCH_SIZE=1
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
DNLNet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Stdc_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}



Convnext_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Dpt_base_params () {
    BATCH_SIZE=1
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
EMANet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
ENCNet_base_params () {
    BATCH_SIZE=1
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
Fastfcn_base_params () {
    BATCH_SIZE=1
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

Fastfscnn_base_params () {
    BATCH_SIZE=2
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
MobileNet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
NonlocalNet_base_params () {
    BATCH_SIZE=2
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}


Knet_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
Mae_base_params () {
    BATCH_SIZE=4
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
Poolformer_base_params () {
    BATCH_SIZE=2
    DATASETS="ADE20K"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}
UPERNet_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

ISANet_base_params () {
    BATCH_SIZE=4
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

HRNet_base_params () {
    BATCH_SIZE=8
    DATASETS="CityScapes"
    DATASETS_DIR="/data/datasets/"
    DATASETS_GPU="/algo/modelzoo/datasets/datasets/"
    CODE_LINK="https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418"

    PYTORCH_VER="16"
    INTEGRATE="mmsegmentation"
    benchmark_mode="True"
}

ShuffleNet_1x_g3_base_params () {
    BATCH_SIZE=512
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets/"
    CODE_LINK="https://github.com/megvii-model/ShuffleNet-Series/tree/aa91feb71b01f28d0b8da3533d20a3edb11b1810"

    PYTORCH_VER="16"
    benchmark_mode="True"
}

PVANet_base_params () {
    BATCH_SIZE=512
    DATASETS="ImageNet2012"
    DATASETS_DIR="/data/pytorch/datasets/"
    CODE_LINK="https://github.com/sanghoon/pytorch_imagenet/tree/e76c2359b3361b6db0f9faf583487b2b1a784808"

    PYTORCH_VER="19"
    benchmark_mode="True"
}


####################
# Docker images info
####################
docker_image_base_params_mlu_pt16 () {
    BASE_IMAGE_NAME="yellow.hub.cambricon.com/pytorch/pytorch:v1.6.0-torch1.6-ubuntu20.04"
    ctr_version="2.5.0"
}

docker_image_base_params_mlu_pt19 () {
    BASE_IMAGE_NAME="yellow.hub.cambricon.com/pytorch/pytorch:v1.6.0-torch1.9-ubuntu20.04"
    ctr_version="2.5.0"
}

docker_image_base_params_gpu_pt16 () {
    BASE_IMAGE_NAME="nvcr.io/nvidia/pytorch:20.07-py3"
    cuda_version="11.0.194"
}

docker_image_base_params_gpu_pt19 () {
    BASE_IMAGE_NAME="nvcr.io/nvidia/pytorch:21.06-py3"
    cuda_version="11.3.1"
}



set_configs () {
    args=$1

    # 获取网络和参数字段
    net=${args%%-*}
    params=${args#*-}

    # 根据每个字段的功能, overide对应参数    
    ddp="False"
    params_array=(${params//-/ })
    for var in ${params_array[@]}
    do
        case "$var" in
            fp32)   ;;
            O[0-3]) precision=$var ;;
            amp)    precision="pyamp" ;;
            mlu)    device="mlu";;
            gpu)    device="gpu" ;;
            ddp)    ddp="True" ;;
            ci_train)  benchmark_mode=False;
                       iters=2;
                       resume_multi_device="True";
                       ;;
            ci_eval)   benchmark_mode=False;
                       iters=2;
                       resume_multi_device="True";
                       evaluate="True";
                       ;;
            *) echo "Unrecognized option: " $var; exit 1;;
        esac
    done

    export dev_type=${device}
    export ddp=${ddp}

    # 调用相应网络的base_params
    ${net}_base_params
    export torch_ver=${PYTORCH_VER}
    if [ $dev_type == "mlu" ];then
        export DATASETS_DIR=${DATASETS_DIR}
        export BATCH_SIZE=${BATCH_SIZE}
        if [ -n "$INIT_CHECKPOINT" ]; then
            export INIT_CHECKPOINT=${INIT_CHECKPOINT}
        else
            export INIT_CHECKPOINT=""
        fi
    else 
        if [ ! -n "$DATASETS_GPU" ]; then  
            export DATASETS_DIR=${DATASETS_DIR}
        else
            export DATASETS_DIR=${DATASETS_GPU}
        fi
        if [ ! -n "$BATCH_SIZE_GPU" ]; then  
            export BATCH_SIZE=${BATCH_SIZE}
        else
            export BATCH_SIZE=${BATCH_SIZE_GPU}
        fi
        if [ -n "$INIT_CHECKPOINT_GPU" ]; then
            export INIT_CHECKPOINT=${INIT_CHECKPOINT_GPU}
        else
            export INIT_CHECKPOINT=""
        fi
    fi
    if [ ! -n "$INTEGRATE" ]; then  
        export NET_FOLDER=${net}
    else
        export NET_FOLDER=${INTEGRATE}
    fi
    # 调用相应网络的base_params
    docker_image_base_params_${dev_type}_pt${PYTORCH_VER}
    export BASE_IMAGE_NAME=${BASE_IMAGE_NAME}
    export cuda_version=${cuda_version}
    export ctr_version=${ctr_version}
    export DATASETS=${DATASETS}

    echo "Docker image: "${BASE_IMAGE_NAME}

    if [[ $benchmark_mode == "True" ]]; then
        ## 检查多卡时是否设置VISIBLE_DEVICES环境变量
        if [[ $ddp == "True" ]]; then
            if [ ! -n "${MLU_VISIBLE_DEVICES}" ] && [ ! -n "${CUDA_VISIBLE_DEVICES}" ] ; then
                echo "Please set env MLU_VISIBLE_DEVICES before running multicards."
                exit 1
            fi
        fi
    fi
}
