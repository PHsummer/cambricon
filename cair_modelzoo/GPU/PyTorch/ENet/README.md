# ENet(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ENet  | PyTorch  | V100/A100  | FP32  | Yes  | Not Tested

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

对于ENet，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:enet-21.06-pytorch-py3
export DATASETS=CityScapes 
export DATASETS_DIR=/algo/modelzoo/datasets/
export CNDB_DIR=/home/zhangboyu/cair_modelzoo/GPU/cndb
export CODE_LINK=https://github.com/davidtvs/PyTorch-ENet/tree/e17d404e2f649a3476eabe39f8a05e5eb77c55fd
export RUN_MODEL_FILE=./PyTorch/ENet/pt-enet.sh
export DOCKERFILE=./PyTorch/ENet/Dockerfile
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行：  
```
bash launch.sh enet 29 8 FP32
```

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:enet-21.06-pytorch-py3
export DATASETS_DIR=/algo/modelzoo/datasets/
export DOCKERFILE=./PyTorch/ENet/Dockerfile
```
### Run
在cair_modelzoo/GPU目录下运行：
```
bash ./PyTorch/ENet/pt-enet-convergency.sh enet
```
精度结果保存在logs文件夹中。

### 精度复现结果
#### Github
Parameter setting:
- Dataset: Cityscapes
- Classes: 19
- Input resolution: 1024x512
- Warmup: 0
- Learning Rate: 0.0005
- Batch size: 4
- Epochs: 300
```
Mean IoU (%): 59.5

```
#### V100自测结果
#### Github
Parameter setting:
- Dataset: Cityscapes
- Classes: 19
- Input resolution: 1024x512
- Warmup: 10
- Learning Rate: 0.0015
- Batch size: 2
- Epochs: 300
```
Mean IoU (%): 59.75
```

### OPEN
ENet使用CrossEntropyLoss做多分类计算，而CrossEntropyLoss结合了LogSoftmax和NLLLoss两个函数。  
由于Pytorch暂不支持NLLLoss设置torch.use_deterministic_algorithms(True)，故在计算至CrossEntropyLoss时将use_deterministic_algorithms设置为False，以跳过固定该算子。  
并未完全固定所有算子，导致每次训练时结果会有所浮动。  
不支持固定的算子列表，请参阅：  
https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

