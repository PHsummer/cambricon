# LiDAR_RCNN(PyTorch)
## 支持情况

Models  | Framework  | Supported GPU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
LiDAR_RCNN  | PyTorch  | V100S  | FP32  | Yes  | Not Tested

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

对于LiDAR RCNN，环境变量设置如下：
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:lidar_rcnn-21.06-pytorch-py3
export DATASETS=mmdet3d_pp 
export DATASETS_DIR=/data1/datasets/autodriver/waymo/preprocessed/
export CNDB_DIR=/home/zhangshuanglin/cair_modelzoo/GPU/cndb
export CODE_LINK=https://github.com/TuSimple/LiDAR_RCNN/tree/89c5fefd46563c15f31b8cda76e7ab09e2ddca0d
export RUN_MODEL_FILE=./PyTorch/LiDAR_RCNN/lidar_rcnn_Performance.sh
export DOCKERFILE=./PyTorch/LiDAR_RCNN/Dockerfile
```

以上数据集路径默认为 10.100.195.22 服务器，如果在10.100.195.17上测试，环境变量应修改为：
```
export DATASETS_DIR=/data/datasets-common/autodriver/waymo_pp/
```

### 参数
launch.sh脚本接受4个参数。
1. model_name：模型名称。
2. batch_size：默认 per card 
3. device_count：训练卡数。
4. precision：数据精度。
### Run
设置好环境变量后，通过以下命令运行: 
```
bash launch.sh lidar_rcnn 256 1 FP32
bash launch.sh lidar_rcnn 256 8 FP32
```

## SOTA精度复现
### 设置环境变量
```
export CONT=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:lidar_rcnn-21.06-pytorch-py3
export DATASETS=mmdet3d_pp
export DATASETS_DIR=/data1/datasets/autodriver/waymo/preprocessed/
export DOCKERFILE=./PyTorch/LiDAR_RCNN/Dockerfile
```

10.100.195.17数据集路径:
```
export DATASETS_DIR=/data/datasets-common/autodriver/waymo_pp/
```

### Run
在cair_modelzoo/GPU目录下运行：
```
bash ./PyTorch/LiDAR_RCNN/lidar_rcnn_Convergency.sh 1
```
读取参数为卡数，默认8卡

该步完成后在容器中LiDAR_RCNN/results文件夹下生成val.bin

最终精度结果需要上传官方服务器查看，具体操作参考https://wiki.cambricon.com/display/Platform/0-+Run中：

四、网络测试 -- 4.2 submission 

因为需要后续步骤, 会保留容器，换执行单卡/8卡需要先删除容器
```
docker container rm -f gpu_golden_bench_lidar_rcnn
```

### 精度复现结果
#### Github
Parameter setting:
- Dataset: mmdet3d_pp
- Classes: Vehicle
- Learning Rate: 0.02
- BATCH_SIZE_PER_GPU: 256
- Epochs: 60

performance on: [Waymo Open Dataset Challenges (3D Detection)](https://waymo.com/open/challenges/2020/3d-detection/)

**多卡**

| Proposals from                                               | Class   |                  |      |
| ------------------------------------------------------------ | ------- | :--------------: | :--: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L1 Vehicle | 75.6 |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L2 Vehicle | 66.8 |

#### V100S自测结果
Parameter setting:
- Dataset: mmdet3d_pp
- Classes: Vehicle
- Learning Rate: 0.02
- BATCH_SIZE_PER_GPU: 256
- Epochs: 60

performance on: [Waymo Open Dataset Challenges (3D Detection)](https://waymo.com/open/challenges/2020/3d-detection/)

**seed未固定**

**8卡**
| Proposals from                                               | Class   |                  |      |
| ------------------------------------------------------------ | ------- | :--------------: | :--: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L1 Vehicle | 75.12 |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L2 Vehicle | 66.60 |

**单卡**
| Proposals from                                               | Class   |                  |      |
| ------------------------------------------------------------ | ------- | :--------------: | :--: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L1 Vehicle | 75.14 |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L2 Vehicle | 66.62 |

**seed固定**

**8卡**
| Proposals from                                               | Class   |                  |      |
| ------------------------------------------------------------ | ------- | :--------------: | :--: |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L1 Vehicle | 75.09 |
| [PointPillars](https://github.com/open-mmlab/mmdetection3d/tree/master/configs/pointpillars) + LiDAR_RCNN | Vehicle | 3D AP L2 Vehicle | 66.59 |
