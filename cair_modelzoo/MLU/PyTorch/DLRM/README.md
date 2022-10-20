# DLRM(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Epochs  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
DLRM  | PyTorch  | MLU370-X8  | FP32  | 1  | 20  | 0.540

## Quick Start Guide
### 数据集
数据集路径为`/data/pytorch/datasets/ml-20mx4x16`。
### Docker镜像
生成DLRM的Docker镜像：`docker build -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:dlrm-catch1.2.1-torch1.6-x86_64-ubuntu18.04 .`
### Run
MLU370-X8单卡FP32测试精度指令：
`bash pt-dlrm-one-X8-from-scratch.sh`