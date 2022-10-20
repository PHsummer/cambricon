# ResNet50(TensorFlow2)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Epochs  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | TensorFlow2  | 370X8  | FP16  | 1  | 90  | Acc@1=0.75668

## Quick Start Guide
### 数据集
数据集放在`/data/tensorflow/training/datasets/ILSVRC2012`目录下。
### Docker镜像
不需要显示修改模型脚本和代码，可以直接使用框架release的docker镜像：tensorflow2-1.10.1-x86_64-ubuntu18.04:latest
### Run
MLU370-X8单卡FP16测试精度指令：
`bash tf2-resnet50-one-X8-from-scratch.sh`