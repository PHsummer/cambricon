# ResNet50-v1.5(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50-v1.5  | PyTorch  | V100/A100  | FP32/TF32/FP16  | Yes  | Not Tested

## Quick Start Guide
GPU Golden Bench通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
### 环境变量
需要设置以下环境变量
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。默认数据集路径为**/data/pytorch/datasets/imagenet_training**，如果使用中数据集在其他位置，请修改**PyTorch/ResNet50v1.5/pt-resnet-50v1.5.sh**文件中数据集相关代码。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**:模型的dockerfile。  
- **CONFIG**:模型的config files所在文件夹，如果对模型的config files有修改，可以将修改后的放到该文件夹，然后在训练模型前替换docker容器中的config files。

对PyTorch ResNet50-v1.5来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:resnet50v1.5-22.01-pytorch-py3
export DATASETS=ImageNet2012 
export DATASETS_DIR=/data
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/NVIDIA/DeepLearningExamples/tree/3a8068b651c8ae919281b638166b3ecfa07d22f5/PyTorch/Classification/ConvNets
export RUN_MODEL_FILE=./PyTorch/ResNet50v1.5/pt-resnet-50v1.5.sh
export DOCKERFILE=./PyTorch/ResNet50v1.5/Dockerfile
export CONFIG=./PyTorch/ResNet50v1.5/configs
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行：
`bash launch.sh resnet50 112 1 FP32`
