# SSD-VGG16(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Iters  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
SSD-VGG16  | PyTorch  | MLU370-X8 | FP16  | 1  |  60000  | Mean AP=0.7750

## Quick Start Guide
### 数据集
数据集路径为`/data/pytorch/datasets//VOCdevkit`。
### Docker镜像
生成SSD-VGG16的Docker镜像：`docker build -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:ssd_vgg16-catch1.2.1-torch1.6-x86_64-ubuntu18.04 .`
### Run
MLU370-X8单卡FP16测试精度指令：
`bash pt-ssd_vgg16-one-X8-from-scratch.sh`
