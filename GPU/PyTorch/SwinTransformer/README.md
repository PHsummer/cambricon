# SwinTransformer(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
SwinTransformer  | PyTorch  | V100/A100  | FP32/TF32/AMP  | Yes  | Not Tested

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

对于Swin Transformer，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:swin-transformer-21.06-pytorch-py3
export DATASETS=ImageNet 
export DATASETS_DIR=/algo/modelzoo/datasets/datasets/imagenet/jpegs
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/microsoft/Swin-Transformer
export RUN_MODEL_FILE=./PyTorch/SwinTransformer/SwinTransformer_Performance.sh
export DOCKERFILE=./PyTorch/SwinTransformer/Dockerfile
```
### 参数
launch.sh脚本接受6个参数。
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
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 1  | bash launch.sh swin_large_patch4_window7_224_22kto1k_finetune 64 1 FP32 false ignore_check
Tesla V100-PCIE-32GB  | PyTorch  | FP16  | 1  | bash launch.sh swin_large_patch4_window7_224_22kto1k_finetune 80 1 AMP false ignore_check
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 8  | bash launch.sh swin_large_patch4_window7_224_22kto1k_finetune 64 8 FP32 false ignore_check
Tesla V100-PCIE-32GB  | PyTorch  | FP16  | 8  | bash launch.sh swin_large_patch4_window7_224_22kto1k_finetune 80 8 AMP false ignore_check
