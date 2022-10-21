# PointPillar(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
PointPillar  | PyTorch  | V100/A100  | FP32/TF32  | Yes  | Not Tested

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

对PyTorch PointPillar来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:pointpillar-21.07-pytorch-py3
export CNDB_DIR=/projs-local/yuanmeng/cair_modelzoo/GPU/cndb
export DOCKERFILE=/projs-local/yuanmeng/cair_modelzoo/GPU/PyTorch/PointPillar/Dockerfile 
export RUN_MODEL_FILE=/projs-local/yuanmeng/cair_modelzoo/GPU/PyTorch/PointPillar/PointPillar_Performance.sh
export DATASETS=nuScenes
export DATASETS_DIR=/data/datasets1/nuscenes_mini
export CODE_LINK=https://github.com/open-mmlab/OpenPCDet
export CONFIG=/projs-local/yuanmeng/cair_modelzoo/GPU/PyTorch/PointPillar
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。

### Run
设置好环境变量后，通过以下命令运行：`bash launch.sh pointpillar 160 1 FP32`

## 精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:pointpillar-22.07-pytorch-py3
export DATASETS_DIR=/data/datasets1/nuScenes
export DOCKERFILE=$YOUR_PATH/cair_modelzoo/GPU/PyTorch/PointPillar/Dockerfile 
export CONFIG_DIR=$YOUR_PATH/cair_modelzoo/GPU/PyTorch/PointPillar
export NVIDIA_TF32_OVERRIDE=0
```
### Run
在`cair_modelzoo/GPU`目录下运行：
`bash ./PyTorch/PointPillar/PointPillar_Convergence.sh`
精度结果保存在logs文件夹中。
#### Github
```
NDS:   0.5823
```

#### GPU默认参数训练精度
```
--------------average performance-------------
trans_err:    0.3391
scale_err:    0.2606
orient_err:   0.3232
vel_err:     0.2991
attr_err:    0.2036
mAP:   0.4411
NDS:   0.5780
```

#### 调参结果
##### changelog

1. Add lr warm up for 1 epoch
2. lr from 0.001 to 0.003
3. batch size per card from 4 to 2



```
--------------average performance-------------
trans_err:    0.3439
scale_err:    0.2609
orient_err:   0.3340
vel_err:     0.2885
attr_err:    0.1991
mAP:   0.4486
NDS:   0.5817

```
