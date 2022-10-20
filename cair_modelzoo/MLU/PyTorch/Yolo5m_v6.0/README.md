# Yolov5m_v6.0(PyTorch)
## 模型概述
`Yolov5m_v6.0`网络是基于 [Yolov5_v6.0](https://github.com/ultralytics/yolov5/tree/v6.0)的m型号，本仓库为其MLU实现。

## 支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- 
Yolov5m_v6.0  | PyTorch  | MLU370-X8  | FP16/FP32  | Yes  | Yes 

## Quick Start Guide
### 数据集
该`Yolov5m_v6.0`脚本基于COCO2017训练，数据集下载：<https://cocodataset.org/#home>。数据集请放在`/data/pytorch/datasets/COCO2017/`目录下(默认MLU服务器该路径数据集已存在)。

### Set up
#### 基于base docker images安装
##### 1、导入镜像
```bash
##下载Cambriocn PyTorch docker镜像
docker load -i xxx.tar.gz
```
##### 2、启动测试容器
```bash
docker run -it --ipc=host -v /data/pytorch/datasets/COCO2017:/data  --device /dev/cambricon_ctl --privileged --name yolo5m_v6.0 xxx
```

##### 3、启动虚拟环境并安装依赖
```bash
source /torch/venv3/pytorch/bin/activate
apt-get update
apt-get -y install libgl1-mesa-glx
pip install -r requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、生成Yolov5m_v6.0的Docker镜像：

```bash
docker build -t yolo5m_v6.0_docker:v1 -f Dockerfile .
```

#####  2、创建容器

```bash
docker run -it --ipc=host -v /data/pytorch/datasets/COCO2017:/data  --device /dev/cambricon_ctl --privileged --name yolo5m_v6.0 yolo5m_v6.0_docker:v1 
```

##### 3、激活环境

```bash
source /torch/venv3/pytorch/bin/activate
```

## 默认参数配置
以下为`Yolov5m_v6.0`模型的默认参数配置：
```bash
lr0: 0.01  # initial learning rate (SGD=1E-2, Adam=1E-3)
lrf: 0.1  # final OneCycleLR learning rate (lr0 * lrf)
momentum: 0.937  # SGD momentum/Adam beta1
weight_decay: 0.0005  # optimizer weight decay 5e-4
warmup_epochs: 3.0  # warmup epochs (fractions ok)
warmup_momentum: 0.8  # warmup initial momentum
warmup_bias_lr: 0.1  # warmup initial bias lr
box: 0.05  # box loss gain
cls: 0.5  # cls loss gain
cls_pw: 1.0  # cls BCELoss positive_weight
obj: 1.0  # obj loss gain (scale with pixels)
obj_pw: 1.0  # obj BCELoss positive_weight
iou_t: 0.20  # IoU training threshold
anchor_t: 4.0  # anchor-multiple threshold
# anchors: 3  # anchors per output layer (0 to ignore)
fl_gamma: 0.0  # focal loss gamma (efficientDet default gamma=1.5)
hsv_h: 0.015  # image HSV-Hue augmentation (fraction)
hsv_s: 0.7  # image HSV-Saturation augmentation (fraction)
hsv_v: 0.4  # image HSV-Value augmentation (fraction)
degrees: 0.0  # image rotation (+/- deg)
translate: 0.1  # image translation (+/- fraction)
scale: 0.5  # image scale (+/- gain)
shear: 0.0  # image shear (+/- deg)
perspective: 0.0  # image perspective (+/- fraction), range 0-0.001
flipud: 0.0  # image flip up-down (probability)
fliplr: 0.5  # image flip left-right (probability)
mosaic: 1.0  # image mosaic (probability)
mixup: 0.1  # image mixup (probability)
copy_paste: 0.0  # segment copy-paste (probability)
```

### 依赖
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.11；
* CNToolKit >=2.8.5-1;
* CNNL >=1.10.5-topkfixnan;
* CNCL >=1.1.2-1;
* CNPyTorch >= 1.3.2;

## 命令行选项

train.py所需参数可于demo.sh中修改，所有可选参数如下：

`python train.py  -h`

```bash
usage: train.py [-h] [--weights WEIGHTS] [--cfg CFG] [--data DATA] [--hyp HYP]
                [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--imgsz IMGSZ]
                [--rect] [--resume [RESUME]] [--nosave] [--noval]
                [--noautoanchor] [--evolve [EVOLVE]] [--bucket BUCKET]
                [--cache [CACHE]] [--image-weights] [--device DEVICE]
                [--multi-scale] [--single-cls] [--adam] [--sync-bn]
                [--workers WORKERS] [--project PROJECT] [--name NAME]
                [--exist-ok] [--quad] [--linear-lr]
                [--label-smoothing LABEL_SMOOTHING] [--patience PATIENCE]
                [--freeze FREEZE] [--save-period SAVE_PERIOD]
                [--local_rank LOCAL_RANK] [--pyamp] [--iters ITERS] [--skip]
                [--entity ENTITY] [--upload_dataset]
                [--bbox_interval BBOX_INTERVAL]
                [--artifact_alias ARTIFACT_ALIAS]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     initial weights path
  --cfg CFG             model.yaml path
  --data DATA           dataset.yaml path
  --hyp HYP             hyperparameters path
  --epochs EPOCHS
  --batch-size BATCH_SIZE
                        total batch size for all GPUs
  --imgsz IMGSZ, --img IMGSZ, --img-size IMGSZ
                        train, val image size (pixels)
  --rect                rectangular training
  --resume [RESUME]     resume most recent training
  --nosave              only save final checkpoint
  --noval               only validate final epoch
  --noautoanchor        disable autoanchor check
  --evolve [EVOLVE]     evolve hyperparameters for x generations
  --bucket BUCKET       gsutil bucket
  --cache [CACHE]       --cache images in "ram" (default) or "disk"
  --image-weights       use weighted image selection for training
  --device DEVICE       cuda device, i.e. 0 or 0,1,2,3 or cpu
  --multi-scale         vary img-size +/- 50%
  --single-cls          train multi-class data as single-class
  --adam                use torch.optim.Adam() optimizer
  --sync-bn             use SyncBatchNorm, only available in DDP mode
  --workers WORKERS     maximum number of dataloader workers
  --project PROJECT     save to project/name
  --name NAME           save to project/name
  --exist-ok            existing project/name ok, do not increment
  --quad                quad dataloader
  --linear-lr           linear LR
  --label-smoothing LABEL_SMOOTHING
                        Label smoothing epsilon
  --patience PATIENCE   EarlyStopping patience (epochs without improvement)
  --freeze FREEZE       Number of layers to freeze. backbone=10, all=24
  --save-period SAVE_PERIOD
                        Save checkpoint every x epochs (disabled if < 1)
  --local_rank LOCAL_RANK
                        DDP parameter, do not modify
  --pyamp               using amp for mixed precision training
  --iters ITERS         Total iters for benchmark.
  --skip                skip val or save pt.
  --entity ENTITY       W&B: Entity
  --upload_dataset      W&B: Upload dataset as artifact table
  --bbox_interval BBOX_INTERVAL
                        W&B: Set bounding-box image logging interval
  --artifact_alias ARTIFACT_ALIAS
                        W&B: Version of dataset artifact to use
```

## Advanced
### Run 
demo.sh 后三个参数依次为 total_batch_size、epoch_num、device_num
### Convergency
Models  | Framework  | MLU   | Data Precision  | Devices  | Batch Size  | Run
----- | ----- | ----- | ----- | ----- | ----- | ----- 
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 16  | 40*16  | bash demo.sh 640 450 16

### Performance
Models  | Framework  | MLU   | Data Precision  | Devices  | Batch Size  | Run
----- | ----- | ----- | ----- | ----- | ----- | -----
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 1  | 70  | bash demo.sh 70 2 1
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 16  | 70*16  | bash demo.sh 1120 2 16

### 训练结果
#### Training accuracy results: MLU370-X8
Models  | Framework  | MLU   | Data Precision  | Devices  | Batch Size | Epochs | AP(iou=0.5)
----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- 
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 16  | 70*16  | 450  | 64.4

#### Training performance results: MLU370-X8

Models  | Framework  | MLU   | Data Precision  | Devices  | Batch Size | Throughput(AMP)  
----- | ----- | ----- | ----- | ----- | ----- | -----
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 1  | 70  | 50.7 fps
Yolov5m_v6.0 | PyTorch  | MLU370-X8  | AMP | 16  | 70*16  | 783.21 fps
