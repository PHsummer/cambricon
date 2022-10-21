# MMClassification
## 支持情况
Models  | Framework  | MLU   |  Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Cifar100_ResNet50  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested
ResNet34  | PyTorch  | MLU290  | INT32  | Yes  | Not Tested
VGG-19  | PyTorch  | MLU290  | INT31  | Yes  | Not Tested
VGG-16-BN   | PyTorch  | MLU290  | INT31   | Yes  | Not Tested
ShuffleNet-V2   | PyTorch  | MLU290  | INT31   | Yes  | Not Tested
ResNetV1D-50 | PyTorch  | MLU290  | INT31   | Yes  | Not Tested
SE-ResNet-50 | PyTorch  | MLU290  | INT31   | Yes  | Not Tested

# Quick Start Guide
## 环境准备
### 使用Dockerfile准备环境
#### 1、生成Docker镜像：
```
docker build . -t yellow.hub.cambricon.com/cair_modelzoo/mlu-mmcls:v1.5.0-torch1.9-ubuntu18.04
```
####  2、创建容器

```
docker run -it --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name mlu_resnet50v1.5 yellow.hub.cambricon.com/cair_modelzoo/mlu-mmcls:v1.5.0-torch1.9-ubuntu18.04 
```
##### 3、启动虚拟环境并安装依赖

```
source /torch/venv3/bin/activate
pip install -r requirement.txt
```

### Run

#### 一键执行训练脚本
Models  | Framework  | MLU   | Data Precision  | Device Num  | Run
----- | ----- | ----- | ----- | ----- | ----- |
Cifar100_ResNet50  | PyTorch  | MLU370-X8  | AMP | 8  | python configs/resnet50_cifar100_16*b496.py 16 --seed 42 
ResNet34  | PyTorch  | MLU290  | INT31  | 8  | python configs/resnet/resnet34_8xb32_in1k.py 8 --seed 42
VGG-19  | PyTorch  | MLU290  | INT31  | 8 | python configs/vgg/vgg19_b32x8_imagenet.py 8 --seed 42
VGG-16-BN   | PyTorch  | MLU290  | INT31   | 8  | python configs/vgg/vgg16_8xb32_in1k.py 8 --seed 42
ShuffleNet-V2   | PyTorch  | MLU290  | INT31   | 16  | python configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py 16 --seed 42
ResNetV1D-50 | PyTorch  | MLU290  | INT31   | 8 | python configs/resnet/resnetv1d50_8xb32_in1k.py8 --seed 42
SE-ResNet-50 | PyTorch  | MLU290  | INT31   | 8  | python configs/seresnet/seresnet50_8xb32_in1k.py 8 --seed 42
