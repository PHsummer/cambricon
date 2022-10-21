# ECAPA-TDNN(PyTorch)

## 支持情况

|   Models   | Framework | Supported GPU | Supported Data Precision | Multi-GPUs | Multi-Nodes |
| :--------: | :-------: | :-----------: | :----------------------: | :--------: | :---------: |
| ECAPA-TDNN |  PyTorch  |     V100      |           FP32           |    Yes     | Not Tested  |

## Quick Start Guide

GPU Golden Bench通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
数据集在10.100.195.22 /data1/pytorch/datasets/voxceleb/voxceleb_wav/wav/中(7000+ id)
debug 可以挂载 10.100.195.22上/ECAPA-TDNN-27.07-pytorch-py3/7dc3e9c7eaa1镜像

### 环境变量

需要设置以下环境变量

- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**:模型的dockerfile。

对ECAPA-TDNN来说，环境变量设置如下：

```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:ECAPA-TDNN-26.07-pytorch-py3
export DATASETS=VoxCeleb12 
export DATASETS_DIR=/data/pytorch/datasets/voxceleb/voxceleb_wav
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/speechbrain/speechbrain/commit/da5701ad81bfc6f56b62fa34835b93c9373c44e3
export RUN_MODEL_FILE=./PyTorch/ECAPA_TDNN/pt_ECAPA_TDNN.sh
export DOCKERFILE=./PyTorch/ECAPA_TDNN/Dockerfile
```

### 参数

launch.sh脚本接受4个参数。

1. model_name：模型名称。
2. batch_size: 批尺寸。
3. device_count：训练卡数。
4. precision：数据精度。

### 文件介绍

dockerFile: 参照 ( <https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/> )
train_ecapa_tdnn.yaml ：模型训练config文件。
modify.patch: 固定了seed

### 数据集

使用 VoxCeleb 数据集 ( <http://www.robots.ox.ac.uk/~vgg/data/voxceleb/> ),参考目录结构如下：

/data/voxceleb/wav/id10526/aCE62x4P_SI/00001.wav

注意要设置DATASETS_DIR

### Run

设置好环境变量后

V100单卡运行命令： `bash launch.sh ECAPA-TDNN 230 1 FP32`

V100八卡运行命令： `bash launch.sh ECAPA-TDNN 230 8 FP32`

**注意**：

device_count设置1为单卡训练，设置其他值为八卡训练；precision目前只支持FP32。

run单卡时候需要修改train_ecapa_tdnn.yaml中batch_size(4->32),learning_rate(0.001->0.008, 0.00000008->0.00000001)这两个参数



