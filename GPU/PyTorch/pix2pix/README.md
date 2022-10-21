# pix2pix(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
pix2pix  | PyTorch  | V100/A100  | FP32/TF32  | Yes  | Not Tested

## Quick Start Guide
GPU Golden Bench通过运行launch.sh启动，通过设置环境变量和提供参数来运行。
### 环境变量
需要设置以下环境变量
- **CONT**：运行网络时的镜像名，通过模型文件夹的Dockerfile生成。
- **DATASETS**：数据集的名称。
- **DATASETS_DIR**：数据集的路径。本模型的数据集**facades**通过dockerfile下载和处理后放在*/workspace/pytorch-CycleGAN-and-pix2pix/datasets/*目录下。
- **CNDB_DIR**：CNDB文件夹路径。
- **CODE_LINK**：模型的代码链接，需要指定commit id。
- **RUN_MODEL_FILE**：模型训练脚本。
- **DOCKERFILE**:模型的dockerfile。  
- **CONFIG**:模型的config files所在文件夹，如果对模型的config files有修改，可以将修改后的放到该文件夹，然后在训练模型前替换docker容器中的config files。

对PyTorch pix2pix来说，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:pix2pix-21.06-pytorch-py3
export DATASETS=facades
export DATASETS_DIR=/data
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py
export RUN_MODEL_FILE=./PyTorch/pix2pix/pt-pix2pix.sh
export DOCKERFILE=./PyTorch/pix2pix/Dockerfile
export CONFIG=./PyTorch/pix2pix/configs
```
### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行单卡测试：

GPU  | Framework | Data Precision   | Cards | Run
----- | ----- | ----- | ----- | -----
Tesla V100-PCIE-32GB  | PyTorch  | FP32 | 1 | bash launch.sh pix2pix_resnet_9blocks 32 1 FP32 false false
Tesla V100-PCIE-32GB  | PyTorch  | FP32 | 8 | bash launch.sh pix2pix_resnet_9blocks 32 8 FP32 false false

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:pix2pix-21.06-pytorch-py3
export DATASETS=facades
export DATASETS_DIR=/data
export CNDB_DIR=./cndb
export CODE_LINK=https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py
export RUN_MODEL_FILE=./PyTorch/pix2pix/pt-pix2pix-convergency.sh
export DOCKERFILE=./PyTorch/pix2pix/Dockerfile
export CONFIG=./PyTorch/pix2pix/configs
```
### Run
在`cair_modelzoo/GPU`目录下运行：

GPU  | Framework | Data Precision   | Cards | Run
----- | ----- | ----- | ----- | -----
Tesla V100-PCIE-32GB  | PyTorch  | FP32 | 1 | bash launch.sh pix2pix_resnet_9blocks 1 1 FP32 false ignore_check

精度结果保存在logs和results文件夹中。
### 精度复现结果
#### [Github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
Github仓库中没有给出精度结果或者loss值,只提供了一些图片测试结果.
#### V100自测
V100在facades数据集上训练200个epochs,第200个epoch的loss如下:
```
(epoch: 200, iters: 100, time: 0.043, data: 0.233) G_GAN: 3.030 G_L1: 16.428 D_real: 0.411 D_fake: 0.096
(epoch: 200, iters: 200, time: 0.044, data: 0.002) G_GAN: 3.474 G_L1: 18.419 D_real: 0.084 D_fake: 0.051
(epoch: 200, iters: 300, time: 0.046, data: 0.002) G_GAN: 2.087 G_L1: 19.271 D_real: 0.028 D_fake: 0.227
(epoch: 200, iters: 400, time: 0.680, data: 0.002) G_GAN: 1.388 G_L1: 19.512 D_real: 0.154 D_fake: 0.433
``` 
取均值loss结果如下表所示:

Model  | G_GAN | G_L1   | D_real | D_fake
----- | ----- | ----- | ----- | -----
pix2pix_resnet_9blocks  | 2.49475  | 18.4075  | 0.16925 | 0.20175

test生成的fake图片和truth图片见"./logs/results"目录.
