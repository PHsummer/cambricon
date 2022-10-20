# BERT-Base(TensorFlow)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Epochs  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
BERT-Base  | TensorFlow  | MLU370-X8  | FP16  | 1  | 2  | acc=0.87196

## Quick Start Guide
### 数据集
数据集放在`/data/tensorflow/training/datasets/Bert`目录下。
### Docker镜像
不需要显示修改模型脚本和代码，可以直接使用框架release的docker镜像：yellow.hub.cambricon.com/tensorflow/tensorflow:v1.10.1-x86_64-ubuntu1804-py3
### Run
MLU370-X8单卡FP16测试精度指令：
`bash tf-bert_base-one-X8-from-scratch.sh`