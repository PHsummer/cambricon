# GraphSAGE(TensorFlow)
## 支持情况

Models  | Framework  | Golden Platforms   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
GraphSAGE | TensorFlow 1.15  | V100 | FP32 | No  | No

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

对TensorFlow GraphSAGE来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:GraphSAGE-22.01-tf1-py3
export DATASETS=ppi
export DATASETS_DIR=/data/tensorflow
export CNDB_DIR=./cndb/
export CODE_LINK=https://github.com/williamleif/GraphSAGE/commit/a0fdef95dca7b456dab01cb35034717c8b6dd017
export RUN_MODEL_FILE=./TensorFlow/Graphsage/tf1-Graphsage.sh
export DOCKERFILE=./TensorFlow/GraphSAGE/Dockerfile
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行：
`bash launch.sh unsupervised_graphsage_mean 512 1 FP32`

注：Dockerfile引入在log日志打印batch_size，检查是否与运行命令的参数一致，方便检查运行错误。
## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:GraphSAGE-22.01-tf1-py3
export DATASETS_DIR=/data/tensorflow
export DOCKERFILE=./TensorFlow/GraphSAGE/Dockerfile
```
### Run
在`cair_modelzoo/GPU`目录下运行：
`bash ./TensorFlow/GraphSAGE/tf1-GraphSAGE-convergency.sh`
精度结果保存在logs文件夹中。
### 精度复现结果
#### Github 
论文给定的 micro-averaged F1 scores 0.486

论文链接：[https://arxiv.org/pdf/1706.02216.pdf]()
#### V100自测
V100 测出 micro-averaged F1 scores 0.486223228666656

由此可达到论文的精度