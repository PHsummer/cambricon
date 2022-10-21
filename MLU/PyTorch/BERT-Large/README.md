# BERT-Large(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Iters  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
BERT-Large  | PyTorch  | MLU370-X8   | FP16  | 1  | 60000  | eval_mlm_accuracy=0.700

## Quick Start Guide
### 数据集
BERT-Large的数据集请放在`/data/pytorch/datasets/bert_data/`目录下。
### Docker镜像
生成BERT-Large的Docker镜像：`docker build -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:bert-large-catch1.1.2-1-x86_64-ubuntu18.04-run .`
### Run
X8单卡AMP训练指令：
`bash pt-bert-large-one-X8-from-scratch.sh`

X8八卡AMP训练指令：
`bash pt-bert-large-8-X8-from-scratch.sh`
### 备注
#### 1.精度
X8单卡训练，lr=1.125e-4下，训练60000步后eval_mlm_accuracy=0.700；
X8八卡训练暂未测试（20220407）。