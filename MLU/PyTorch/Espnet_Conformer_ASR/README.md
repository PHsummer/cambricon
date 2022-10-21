# Espnet Conformer

## 模型概述

代码来源Espnet：https://github.com/espnet/espnet/tree/master/egs2/aishell/asr1

Espnet conformer是一个自动语音识别模型（Auto Speech Recognition，ASR）。此模型使用conformer模块作为encoder，transformer模块作为decoder，在Aishell数据集上可达到准确率95.4，CER 4.6的精度。

Conformer模块结果可参考https://paperswithcode.com/paper/conformer-convolution-augmented-transformer

## 支持情况

| Models | Framework | Supported MLU | Supported Data Precision | Multi-MLUs | Multi-Nodes | 
|----|----|----|----|----|----|
| Conformer | Pytorch | MLU370-X8 | FP32/AMP | Yes | Not tested |

## 结果展示

| device | DDP | precision | global batch bins | epoch | result (test Corr) | time to train | performance |
|----|----|----|----|----|----|----|----|
|MLU370-X8 | 16 | AMP | 96e6 | 100 | 95.4 | 15h | 116,930,572 bins/s | 

## Quick Start Guide

### 数据集

Aishell数据集下载地址：http://www.aishelltech.com/kysjcp

数据集路径 `/data/datasets/aishell`

数据集文件树

```
aishell
├── data_aishell.tar
└── resource_aishell.tar
```
(程序会自动解压数据集，在此文件夹中生成解压后的文件)

### 运行

#### 方式一、从docker镜像运行

`run_docker_performance_test.sh`和`run_docker_convergence_test.sh`脚本可自动创建docker镜像，在容器中进行模型训练，并计算训练性能。

##### 1. 修改模型训练参数

训练卡数默认为16卡。若需要修改，需要在脚本中`docker exec`指令中修改`export MLU_VISIBLE_DEVICE`和`--ngpu`参数。

例：使用4卡训练

```shell
docker exec -ti --use-nas-user ${CONT_NAME} /bin/bash -c \
        "source /torch/venv3/pytorch/bin/activate && \
        export PYTHONPATH=\$PYTHONPATH:/test_espnet/espnet_mlu && \
        export PATH=\$PATH:/test_espnet/espnet_mlu/tools/SCTK/bin && \
        export NLTK_DATA=/test_espnet/espnet_mlu/nltk_data && \
        export MLU_VISIBLE_DEVICES=0,1,2,3 && \
        ./run.sh --ngpu 4"
```

其他模型训练参数（如batch bins，max epoch，use amp）保存在espnet_config/train_asr_conformer_mlu.yaml中。若需要调整参数，需修改参数文件中对应项的数值。

##### 2. 修改docker挂载路径

`run_docker_performance_test.sh`和`run_docker_convergence_test.sh`需要配置运行docker时挂载目录的路径。

AISHELL_DIR：Aishell数据集的根目录路径。

OUTPUT_DIR：模型输出checkpoint及训练日志的路径。

CACHE_DIR：模型保存预处理数据的路径。

以上三个路径均需要当前用户的读写权限，CACHE_DIR需要约22G的空间。

##### 3.通过脚本运行

若要测试模型收敛性，运行`run_docker_convergence_test.sh`。此脚本会进行完整的训练过程。

若要测试模型性能，运行`run_docker_performance_test.sh`。此脚本仅进行200 step的训练，并会跳过eval阶段，可以更快得到性能数据。

#### 方式二、搭建本地环境运行

##### 1. 搭建Python环境

需要python >= 3.6，torch >= 1.6 (torch需要支持mlu)

使用 `pip install -r requirments.txt`安装其他依赖。

##### 2. 安装bc、sox命令

可使用`apt-get install bc sox -y`或其他方式安装。

##### 3. 安装kaldi、kenlm、SCTK工具包

将kaldi、kenlm、SCTK工具包安装在tools文件夹下，具体安装指令方式参考以下github链接。

kaldi：https://github.com/kaldi-asr/kaldi

kenlm: https://github.com/kpu/kenlm

SCTK: https://github.com/usnistgov/SCTK

##### 4.运行模型

`cd espnet_mlu/egs2/aishell/asr1`

与在docker中运行类似，需要配置模型训练参数，再运行测试脚本。

本地环境中模型训练参数保存在`conf`文件夹，需要在`run.sh`设置使用的配置文件的路径。

测试性能运行`run_performance_test.sh`.

测试收敛性运行`run_convergence_test.sh`.

## 默认训练参数

默认训练参数配置文件保存在`espnet_config/train_asr_conformer_mlu.yaml`.

| 参数        | 参数名     | 默认值   |
| ----------- | ---------- | -------- |
| batch大小   | batch_bins | 96000000 |
| 训练周期数  | max_epoch  | 100      |
| 是否使用AMP | use_amp    | true     |
|学习率|lr|0.001|
|学习率衰减|weight_decay|0.0000005|
|学习率warmup时间|warmup_steps|20000|
|子线程数|num_workers|8|
## 性能计算方式

模型的性能计算在run_with_docker.sh末尾。训练结束后通过抓取OUTPUT_DIR下lm和asr的train.log中的数据分别进行计算。

| 性能参数| 计算方式 |
|----|----|
| 吞吐量 | batch_bins/train_time. 其中train_time为log中每个iter的e2e时间均值。 |
| 训练总时间 | log中的elapsed time，是运行python训练文件的总时间。 |
| 准确率 | RESULT.md中的CER Test Corr |
