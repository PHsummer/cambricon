# CenterNet(PyTorch)
## 支持情况

Models  | Framework  | Supported Device   | Supported Data Precision  | Multi-Devices  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
CenterNet  | PyTorch  | MLU370-X8/GPU  | FP32  | No  | Not Tested

## Quick Start Guide
Model Zoo通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
### 环境变量
需要设置以下环境变量
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**：模型的dockerfile。
- **CONFID_DIR**：模型config文件。  

对PyTorch CenterNet来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/mlu_benchmark:CenterNet-ctr24-pytorch-py3
export CNDB_DIR=${YOUR_PATH}/cair_modelzoo/MLU/Validated_Models/tools/cndb
export DOCKERFILE=${YOUR_PATH}/cair_modelzoo/MLU/Validated_Models/PyTorch/CenterNet/Dockerfile 
export RUN_MODEL_FILE=${YOUR_PATH}/cair_modelzoo/MLU/Validated_Models/PyTorch/CenterNet/CenterNet_Performance.sh
export DATASETS=COCO2017
export DATASETS_DIR=/data/pytorch/datasets/COCO2017
export CODE_LINK=https://github.com/xingyizhou/CenterNet
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
5. database：结果上传数据库
    - mlu-demo：结果上传到demo-db数据库；
    - mlu-validated：结果上传到mlu-validated数据库；
    - 默认为false：不上传结果到数据库。
6. ignore_check: 该参数控制检查MLU Performance状态，测试机器独占下的性能。
    - 默认参数为false：强制检查Performance测试环境，如果检查不过，程序中止，launch.sh脚本不会继续运行。
    - 参数设置为ignore_check
    
### Run
设置好环境变量后，通过以下命令运行：


Device  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | bash test_detection.sh 0 fp32-mlu
GPU  | PyTorch  | FP32  | 1  | bash test_detection.sh 0 fp32-gpu
