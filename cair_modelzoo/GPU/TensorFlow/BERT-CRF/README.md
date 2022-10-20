# BERT-CRF(TensorFlow)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
BERT-CRF  | TensorFlow  | V100/A100  | FP32/TF32 | No  | No

## Quick Start Guide
GPU Golden Bench通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
### 环境变量
需要设置以下环境变量
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**:模型的dockerfile。  

对TensorFlow BERT-CRF来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:bert-crf-22.01-tf1-py3
export DATASETS=CoNLL-2003
export DATASETS_DIR=/data
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/kyzhouhzau/BERT-NER/tree/0f77e478872453df51cd3c65d1a39b12d9617f9d
export RUN_MODEL_FILE=./TensorFlow/BERT-CRF/tf1-bert-crf.sh
export DOCKERFILE=./TensorFlow/BERT-CRF/Dockerfile
```
**注意**：BERT-CRF需要`cased_L-12_H-768_A-12`模型文件，可以从[这里](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)下载，解压后放在`./TensorFlow/BERT-CRF/`目录下，在build docker镜像的时候会COPY进去。**如果不放进去，会对精度复现产生影响！**
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行：
`bash launch.sh BERT-CRF 32 1 FP32`

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:bert-crf-22.01-tf1-py3
export DATASETS=CoNLL-2003
export DATASETS_DIR=/data
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/kyzhouhzau/BERT-NER/tree/0f77e478872453df51cd3c65d1a39b12d9617f9d
export RUN_MODEL_FILE=./TensorFlow/BERT-CRF/tf1-bert-crf-convergency.sh
export DOCKERFILE=./TensorFlow/BERT-CRF/Dockerfile
```
### Run
在`cair_modelzoo/GPU`目录下运行：
`bash launch.sh BERT-CRF 32 1 FP32`
**注意**：在此模型中,上面传入的batch_size和device_count不起作用,精度测试只支持*batch_size=32*和*device_count=1*的测试结果.
精度结果保存在logs文件夹中。
### 精度复现结果
#### Github 
Parameter setting:
* do_lower_case=False
* num_train_epochs=4.0
* crf=False

```
accuracy:  98.15%; precision:  90.61%; recall:  88.85%; FB1:  89.72
              LOC: precision:  91.93%; recall:  91.79%; FB1:  91.86  1387
             MISC: precision:  83.83%; recall:  78.43%; FB1:  81.04  668
              ORG: precision:  87.83%; recall:  85.18%; FB1:  86.48  1191
              PER: precision:  95.19%; recall:  94.83%; FB1:  95.01  1311
```
#### V100自测结果
Parameter setting:
* do_lower_case=False
* num_train_epochs=4.0
* crf=True
```
accuracy:  98.16%; precision:  88.95%; recall:  90.65%; FB1:  89.79
              LOC: precision:  92.19%; recall:  92.72%; FB1:  92.45  1395
             MISC: precision:  78.88%; recall:  82.19%; FB1:  80.50  696
              ORG: precision:  84.19%; recall:  88.50%; FB1:  86.29  1252
              PER: precision:  95.47%; recall:  94.74%; FB1:  95.10  1301
``` 
