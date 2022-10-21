# GLM(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Iters  | loss
----- | ----- | ----- | ----- | ----- | ----- | ----- |
GLM-base | DeepSpeed | MLU370-X8   | FP16  | 1  | 1000  | loss= 5.573
GLM-XXLarge | DeepSpeed | MLU370-X8   | FP16  | 16  | 1000  | loss= 5.707

## Quick Start Guide

### 数据集
GLM 的数据集放在`/data/cpm/glm/`目录下。

### Docker镜像
生成 GLM 的 Docker 镜像：`docker build --network=host -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:glm-catch1.3.0-x86_64-ubuntu18.04 .`

### Run
GLM-base 在X8八卡AMP训练指令：
`bash scripts/ds_pretrain_mlu.sh`

GLM-XXLarge 在X8八卡AMP训练指令：
`bash scripts/ds_pretrain_mlu_10B.sh`


### 备注

#### 安装 IB 通信：
只有 GLM-XXLarge 模型在多机运行且使能IB通信时才需要安装IB驱动，因此Dockerfile中保留安装IB驱动的方法，但默认不安装IB驱动。
如果用户需要IB通信， 需用户去当前目录的Dockerfile中解除安装IB驱动的注释。

#### GLM-XXLarge 训练的注意事项：
GLM-XXLarge 模型的参数量是百亿规模，在单台 X8 机器上无法运行，至少需要两台 X8 机器，因此在运行前需要配置环境,
具体步骤参见：GLM/doc/test_reports_20220331/glm-report-20220331.pdf 文件中介绍的第5节`复现方法`。
