# GPU Golden Bench
GPU Golden Bench是为了解决产品线统一竞品对标性能数据这一目的，其次在执行的过程中希望能做到尽量自动化更新，减少人工介入。相关介绍详见[GPU Golden Bench](http://wiki.cambricon.com/display/Platform/GPU+Golden+Bench)。
## Overview
Repo中重要的文件和文件夹介绍如下：
- launch.sh：主程序，负责接收、检查和处理环境变量和参数，获取软、硬件信息，检查性能相关的设置项，启动docker环境训练，处理结果数据和上传到数据库。
- PyTorch：Pytorch模型文件夹，每个模型文件夹下面包含一个README、Dockerfile、和shell脚本。
    - Dockerfile：指定该模型运行的Docker镜像，codebase，分支或者commit id。
    - shell脚本：在docker容器中执行训练指令，获得训练结果，并生成benchmark_log。
    - README.md
    - configs：可选。对模型configs的修改或其他patch文件会放入其中。
- Tensorflow和Tensorflow2文件夹与PyTorch文件夹结构一致。
- dump_for_cndb.py：将benchmark_log数据处理成CNDB可以处理的yaml格式。
- cndb：CNDB库，将训练环境和训练结果上传到数据库的工具，原始链接为[CNDB](http://gitlab.software.cambricon.com/liangfan/cndb)，cair_modelzoo中对部分代码进行了修改以支持GPU使用。
- gpu_software_info.json：Docker镜像中的软件信息，经过dump_for_cndb.py处理后上传数据库。可将自制镜像上传到公司黄区[Harbor仓库]（http://yellow.hub.cambricon.com/harbor/projects/9/repositories/gpu_golden_bench），然后将镜像名和软件信息更新到此json文件中。
- tools
    - check_gpu_perf.sh:检查GPU训练Performance相关设置是否满足。
    - set_gpu_perf.sh：设置GPU训练Performance参数。
## Setup
请按照以下步骤安装相关依赖。
### Requirements
- python >= 3.6 ((3.6.13上已测试))
### Install
1. `git clone http://gitlab.software.cambricon.com/neuware/platform/cair_modelzoo.git && git checkout gpu_golden_bench`
2. 安装**CNDB**: `cd cair_modelzoo/GPU/cndb && pip install -r requirements.txt && python setup.py install`
## Quick Start Guide
下面以`PyTorch ResNet50-v1.5`来介绍GPU Golden Bench的使用，不同模型的使用请查看具体模型的README.md。
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
- **CONFIG**：模型的config文件夹，对模型的configs修改或其他patch文件放到此文件夹，在模型训练脚本中使用。

对PyTorch ResNet50-v1.5来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:resnet50v1.5-22.01-pytorch-py3
export DATASETS=ImageNet2012 
export DATASETS_DIR=/data
export CNDB_DIR=/home/limengxiao/cair_modelzoo/GPU/cndb
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
## 结果展示
launch.sh正常运行后，默认将中间文件保存在`./results`中，训练打印log保存在`./logs`文件夹中，每次运行后默认将`./results`删除，避免结果反复处理和上传。  
默认将数据上传到`cndb_demo`数据库中，可在[Demo Dashboard](http://dataview.cambricon.com/superset/dashboard/86/)上查看。
