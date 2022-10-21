# MacBert(PyTorch)
## 支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
MacBert  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested

## Quick Start Guide
Model Zoo通过运行launch.sh启动，通过设置环境变量和提供参数来运行。   
### 环境变量
需要设置以下环境变量
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。
- **INIT_CHECKPOINT**: 需加在的预训练模型路径，不需要可以不填。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**：模型的dockerfile。
- **CONFID_DIR**：模型config文件。  

对PyTorch MacBert来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/mlu_benchmark:MacBert-ctr23-pytorch-py3
export CNDB_DIR=./tools/cndb
export DOCKERFILE=./PyTorch/LanguageModeling/MacBert/Dockerfile 
export RUN_MODEL_FILE=./PyTorch/LanguageModeling/MacBert/MacBert-Performance.sh
export DATASETS=DRCD
export DATASETS_DIR=/data/datasets/DRCD
export INIT_CHECKPOINT=/projs/platform/public/models
export CODE_LINK=https://github.com/huggingface/transformers
```
### 参数
launch.sh脚本接受6个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
5. database：结果上传数据库
    - mlu-demo：结果上传到MLU-demo数据库；
    - mlu-validated：结果上传到MLU-Validated数据库；
    - 默认为false：不上传结果到数据库。
6. ignore_check: 该参数控制检查MLU Performance状态，测试机器独占下的性能。
    - 默认参数为false：强制检查Performance测试环境，如果检查不过，程序中止，launch.sh脚本不会继续运行。
    - 参数设置为ignore_check
    
### Run
设置好环境变量后，通过以下命令运行：


MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_languagemodeling.sh 1 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  | export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_languagemodeling.sh 1 fp32-mlu-ddp
V100  | PyTorch  | FP32  | 1  | ./test_languagemodeling.sh 1 fp32-gpu
V100  | PyTorch  | FP32  | 8  | export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_languagemodeling.sh 1 fp32-gpu-ddp
