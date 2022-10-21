# Tacotron2(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Iters  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
Tacotron2  | PyTorch  | MLU370-X8   | FP16  | 1  | 1500  | train_loss=0.287

## Quick Start Guide
### 数据集
数据集路径为`/data/pytorch/datasets/TTS`。
### Docker镜像
生成Tacotron2的Docker镜像：`docker build -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:tacotron2-catch1.2.1-torch1.9-x86_64-ubuntu18.04 .`
### Run
MLU370-X8单卡FP16测试精度指令：
`bash pt-tacotron2-one-X8-from-scratch.sh`
**当前batch size为128，可能引起MLU内存不够报错，报错后检查是否独占。PyTorch组的精度测试使用的batch size为96，可以修改“scratch_tacotron2_2mlu_AMP.sh”中batch size为96，先删除docker容器，再删除镜像，然后按照README从头操作。**