# MLU Validated Models
本仓库汇总MLU上已经摸排过的模型，方便市场和FAE获取完整网络信息，减少后续模型摸排工作量。
# Overview
Repo中重要的文件和文件夹介绍如下：
- launch.sh：主程序，负责接收、检查和处理环境变量和参数，获取软、硬件信息，检查性能相关的设置项，启动docker环境训练，处理结果数据和上传到数据库。
- PyTorch：Pytorch模型文件夹，每个模型文件夹下面包含一个README、Dockerfile、shell脚本、configs。
    - README.md
    - Dockerfile：指定该模型运行的Docker镜像，codebase，分支或者commit id。
    - shell脚本：在docker容器中执行训练指令，获得训练结果，并生成mlu_benchmark_log。
    - configs：可选。对模型configs的修改或其他patch文件(对model或者train的修改)会放入其中。
- Tensorflow和Tensorflow2文件夹与PyTorch文件夹结构一致。
- tools
    - cndb：CNDB库，将训练环境和训练结果上传到数据库的工具，原始链接为[CNDB](http://gitlab.software.cambricon.com/liangfan/cndb)，注意此cndb文件夹和GPU Golden Bench的cndb文件夹代码不完全相同。
    - check_mlu_perf.sh:检查MLU训练Performance相关设置是否满足。
    - set_mlu_perf.sh：设置MLU训练Performance参数。    
    - dump.py：将mlu_benchmark_log数据处理成CNDB可以上传的yaml格式。
    - soft_info.json：框架Docker镜像中的软件信息，包括CTR版本、框架版本、发布日期等信息。

## RUN
以分类网络为例:
1. 进入对应任务的文件夹
```
cd Validated_Models/PyTorch/Classification
```

2. 运行脚本

```
#单卡:
test_ classify.sh <model_id>   <data precision>-<device>

#多卡:

export MLU_VISIBLE_DEVICES={Device_num} && ./test_classify.sh <model_id>   <data precision>-mlu-ddp
export CUDA_VISIBLE_DEVICES={Device_num} && ./test_classify.sh <model_id>   <data precision>-gpu-ddp
```

## 结果展示

可在[Dashboard](http://dataview.cambricon.com/superset/dashboard/mlu_validated_models/)上查看validated model zoo结果。


# 支持模型列表
### Image Classification<a name="classification"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [ECANet](PyTorch/Classification/ECANet) | PyTorch | Training | ImageNet2012 | [ECANet](PyTorch/Classification/ECANet) | [ECANet_code](https://github.com/BangguWu/ECANet) | |
| [InceptionV4](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [InceptionV4](PyTorch/Classification/timm) | [InceptionV4_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_v4.py) | |
| [Inception_ResNetv2](PyTorch/Classification/Inception_ResNetv2) | PyTorch | Training | ImageNet2012 | [Inception_ResNetv2](PyTorch/Classification/Inception_ResNetv2) | [Inception_ResNetv2_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/inception_resnet_v2.py) | |
| [MobileNetV3](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [MobileNetV3](PyTorch/Classification/timm) | [MobileNetV3_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py) | |
| [ResNeXt](PyTorch/Classification/ResNeXt) | PyTorch | Training | ImageNet2012 | [ResNeXt](PyTorch/Classification/ResNeXt) | [ResNeXt_torchvison](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) | |
| [ShuffleNetV2](PyTorch/Classification/ShuffleNetV2) | PyTorch | Training | ImageNet2012 | [ShuffleNetV2](PyTorch/Classification/ShuffleNetV2) | [ShuffleNetV2_torchvison](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py) | |
| [SqueezeNet](PyTorch/Classification/SqueezeNet) | PyTorch | Training | ImageNet2012 | [SqueezeNet](PyTorch/Classification/SqueezeNet) | [SqueezeNet_torchvison](https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py) | |
| [WRN50-v2](PyTorch/Classification/WRN50-v2) | PyTorch | Training | ImageNet2012 | [WRN50-v2](PyTorch/Classification/WRN50-v2) | [WRN50-v2_torchvison](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py) | |
| [Xception](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [Xception](PyTorch/Classification/timm) | [Xception_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/xception.py) | |
| [DPN68](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [DPN68](PyTorch/Classification/timm) | [DPN68_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/dpn.py) | |
| [vovnet](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [vovnet](PyTorch/Classification/timm) | [vovnet_timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vovnet.py) | |
| [HRNet](PyTorch/Classification/timm) | PyTorch | Training | ImageNet2012 | [HRNet](PyTorch/Classification/timm) | [HRNet_timm](https://github.com/rwightman/pytorch-image-model) | |
| [ShuffleNet_1x_g3](PyTorch/Classification/ShuffleNet_1x_g3) | PyTorch | Training | ImageNet2012 | [ShuffleNet_1x_g3](PyTorch/Classification/ShuffleNet_1x_g3) | [ShuffleNet-Series](https://github.com/megvii-model/ShuffleNet-Series/blob/master/ShuffleNetV1/train.py) | |

### Detection<a name="Detection"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [CenterNet](PyTorch/Detection/CenterNet) | PyTorch | Training | COCO2017 | [CenterNet](PyTorch/Classification/CenterNet) | [CenterNet_code](https://github.com/xingyizhou/CenterNet) | |
| [FaceBox](PyTorch/Detection/FaceBox) | PyTorch | Training | WIDER_FACE | [FaceBox](PyTorch/Classification/FaceBox) | [FaceBox_code](https://github.com/zisianw/FaceBoxes.PyTorch) | |
| [MobileNetV2_SSDlite](PyTorch/Detection/mmdetection) | PyTorch | Training | COCO2017 | [MobileNetV2_SSDlite](PyTorch/Detection/mmdetection) | [MobileNetV2_SSDlite_mmdet](https://github.com/open-mmlab/mmdetection/blob/ca11860f4f3c3ca2ce8340e2686eeaec05b29111/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py) | |
| [PVANet](PyTorch/Detection/PVANet) | PyTorch | Training | ImageNet2012 | [PVANet](PyTorch/Detection/PVANet) | [PVANet_code](https://github.com/sanghoon/pytorch_imagenet/blob/master/train_imagenet.py) | |


### ActionRecognition<a name="ActionRecognition"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [ST-GCN](PyTorch/ActionRecognition/ST-GCN) | PyTorch | Training | COCO2017 | [ST-GCN](PyTorch/ActionRecognition/ST-GCN) | [ST-GCN_code](https://github.com/fendou201398/st-gcn/tree/91e4046fe2274ac74d6220998996cdcd955ba715) | |

### LanguageModeling<a name="LanguageModeling"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [Bert](PyTorch/LanguageModeling/Bert) | PyTorch | Training | squad| [ST-GCN](PyTorch/LanguageModeling/Bert) | [Bert_hugingface](https://github.com/huggingface/transformers) | |
| [MacBert](PyTorch/LanguageModeling/MacBert) | PyTorch | Training | DRCD| [MacBert](PyTorch/LanguageModeling/MacBert) | [MacBert_hugingface](https://github.com/huggingface/transformers) | |
| [RoBERTa](PyTorch/LanguageModeling/RoBERTa) | PyTorch | Training | squad| [RoBERTa](PyTorch/LanguageModeling/RoBERTa) | [RoBERTa_hugingface](https://github.com/huggingface/transformers) | |
| [KG_Bert](PyTorch/LanguageModeling/KG_Bert) | PyTorch | Training | WN11| [KG_Bert](PyTorch/LanguageModeling/KG_Bert) | [KG_Bert_code](https://github.com/yao8839836/kg-bert) | |

### Segmentation<a name="Segmentation"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [ERFNet](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | CityScapes | [ERFNet](PyTorch/Segmentation/mmsegmentation) | [ERFNet_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/erfnet) | |
| [ICNet](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | CityScapes | [ICNet](PyTorch/Segmentation/mmsegmentation) | [ICNet_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/icnet) | |
| [SegFormer](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [SegFormer](PyTorch/Segmentation/mmsegmentation) | [SegFormer_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/segforme) | |
| [Twins](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [Twins](PyTorch/Segmentation/mmsegmentation) | [Twins_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/twins) | |
| [FCN](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [FCN](PyTorch/Segmentation/mmsegmentation) | [FCN_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/fcn) | |
| [PSPNet](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [PSPNet](PyTorch/Segmentation/mmsegmentation) | [PSPNet_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/pspnet) | |
| [BiSeNet](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [BiSeNet](PyTorch/Segmentation/mmsegmentation) | [BiSeNet_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/bisenetv1) | |
| [FCN_R50](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [FCN_R50](PyTorch/Segmentation/mmsegmentation) | [FCN_R50_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/fcn) | |
| [DeeplabV3_R50](PyTorch/Segmentation/mmsegmentation) | PyTorch | Training | ADE20K | [DeeplabV3_R50](PyTorch/Segmentation/mmsegmentation) | [DeeplabV3_R50_mmseg](https://github.com/open-mmlab/mmsegmentation/tree/4d0eb367e9136c0000a5ee9ee45de1db3a557418/configs/deeplabv3) | |
### Tracking<a name="Tracking"></a>
| 网络 | 版本 | 项目 | 数据集 | 模型路径 | 模型来源 | 备注 |
------|------|------|-------|---------|--------|----------|
| [GOTURN](PyTorch/Tracking/GOTURN) | PyTorch | Training | ILSVRC2014_DET & Alov300++ | [GOTURN](PyTorch/Segmentation/GOTURN) | [GOTURN_code](https://github.com/amoudgl/pygoturn/tree/1785ca7106fc0aa9e0fd03b440f34454e8a10e78) | |
| [MDNet](PyTorch/Tracking/MDNet) | PyTorch | Training | ILSVRC2015 | [MDNet](PyTorch/Segmentation/MDNet) | [MDNet_code](https://github.com/hyeonseobnam/py-MDNet/tree/680fa4d58c427b8b647fa59dd9cd61a9fb7061f6) | |
| [Siamese](PyTorch/Tracking/Siamese) | PyTorch | Training | omniglot | [Siamese](PyTorch/Segmentation/Siamese) | [Siamese_code](https://github.com/fangpin/siamese-pytorch/tree/5543f1e844964b116dc9d347a5eb164c6a7afe6d) | |
