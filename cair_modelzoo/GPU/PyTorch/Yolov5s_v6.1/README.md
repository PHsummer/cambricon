# Yolov5s_v6.1(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Yolov5s_v6.1  | PyTorch  | V100  | FP32  | Yes  | Not Tested

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
- **CONFID_DIR**:模型config文件。  

对PyTorch Yolov5s_v6.1来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:Yolov5s_v6.1-21.06-pytorch-py3
export CNDB_DIR=${YOUR_PATH}/cair_modelzoo/GPU/cndb
export DOCKERFILE=${YOUR_PATH}/cair_modelzoo/GPU/PyTorch/Yolov5s_v6.1/Dockerfile 
export RUN_MODEL_FILE=${YOUR_PATH}/cair_modelzoo/GPU/PyTorch/Yolov5s_v6.1/Yolov5s_v6.1_Performance.sh
export DATASETS=COCO2017
export DATASETS_DIR=/data/datasets-common/COCO2017/
export CODE_LINK=https://github.com/open-mmlab/OpenPCDet
export CONFIG=${YOUR_PATH}/cair_modelzoo/GPU/PyTorch/Yolov5s_v6.1
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
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 1  | bash launch.sh Yolov5s_v6.1 128 1 FP32 false ignore_check
Tesla V100-PCIE-32GB  | PyTorch  | FP32  | 8  | bash launch.sh Yolov5s_v6.1 1344 8 FP32 false ignore_check

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:Yolov5s_v6.1-21.06-pytorch-py3
export DATASETS=COCO2017
export DATASETS_DIR=/data/datasets-common/COCO2017
export DOCKERFILE=./PyTorch/Yolov5s_v6.1/Dockerfile
```
### Run
在cair_modelzoo/GPU目录下运行:
```
bash ./PyTorch/Yolov5s_v6.1/Yolov5s_v6.1_Convergency.sh
```
精度结果保存在logs文件夹中。

### 精度复现结果
### 单卡
#### Github
Parameter setting:
- Dataset: COCO2017
- Image size: 640
- Batch size(single card): 128
- Weight decay: 0.001
- Epochs: 300
```
iou0.5 mAP: 56.8

```
#### V100自测结果
Parameter setting:
- Dataset: COCO2017
- Image size: 640
- Batch size(single card): 128
- Weight decay: 0.001
- Epochs: 550
```
iou0.5 mAP: 57.3 (Yolov5s_v6.1_1card_bs128_v100.log epoch:521)

```
### 多卡
#### Github
无官方数据

#### V100自测结果
Parameter setting:
- Dataset: COCO2017
- Image size: 640
- Batch size(8 cards): 1344
- Weight decay: 0.001
- Epochs: 300
```
iou0.5 mAP: 54.2 (Yolov5s_v6.1_8cards_bs1344_v100.log epoch:292)
