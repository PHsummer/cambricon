# ATSS-ResNet50(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ATSS-ResNet50  | PyTorch  | V100/A100  | FP32/TF32  | Yes  | Not Tested

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

对于ATSS-ResNet50，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:atss-r50-21.06-pytorch-py3
export DATASETS=COCO2017 
export DATASETS_DIR=/algo/modelzoo/datasets/datasets
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/open-mmlab/mmdetection/tree/ca11860f4f3c3ca2ce8340e2686eeaec05b29111
export RUN_MODEL_FILE=./PyTorch/ATSS-ResNet50/pt-atss-r50.sh
export DOCKERFILE=./PyTorch/ATSS-ResNet50/Dockerfile
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。  
5. database：结果上传到哪个数据库。参数目前有如下选项  
    gpu-demo：结果上传到demo-db数据库； 
    gpu-performance：结果上传到gpu-golden-bench数据库；  
    gpu-convergency：结果上传到gpu-golden-bench-convergency数据库；  
    默认为false：不上传结果到数据库。  
6. ignore_check: 该参数控制检查GPU Performance状态，测试机器独占下的性能。  
    默认参数为false：强制检查Performance测试环境，如果检查不过，程序中止，launch.sh脚本不会继续运行。
    参数设置为ignore_check：例如”bash launch.sh resnet50 112 1 FP32 ignore_check"，会打印出检查不通过的信息，但程序会继续执行，建议确认环境无误后才添加”ignore_check“参数！
### Run
设置好环境变量后，通过以下命令运行：  

GPU  | Framework  | Data Precision   | Cards  | Run
----- | ----- | ----- | ----- | ----- |
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 1  | bash launch.sh ATSS-R50 19 1 FP32 false ignore_check
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 8  | bash launch.sh ATSS-R50 19 8 FP32 false ignore_check
  

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:atss-r50-21.06-pytorch-py3
export DATASETS=COCO2017
export DATASETS_DIR=/algo/modelzoo/datasets/datasets
export DOCKERFILE=./PyTorch/ATSS-ResNet50/Dockerfile
```
### Run
在cair_modelzoo/GPU目录下运行：
```
bash ./PyTorch/ATSS-ResNet50/pt-atss-r50-convergency.sh
```
精度结果保存在logs文件夹中。

### 精度复现结果
#### Github
Parameter setting:
- Dataset: COCO2017
- Warmup: Unknow
- Batch size(per card): 2
- Epochs: 12
```
box AP: 39.4

```
#### V100自测结果
#### Github
Parameter setting:
- Dataset: COCO2017
- Warmup: 500 iter
- Batch size(per card): 2
- Epochs: 12
```
box AP: 39.5

```
