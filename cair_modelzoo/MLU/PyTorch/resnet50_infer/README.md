# ResNet50-v1.5(PyTorch)
## **模型概述**
ResNet50-v1.5网络是基于 [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)的改进版本，该模型结构源于[Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf)。

该仓库为TorchVision ResNet50V1.5的MLU实现。ResNet50V1.5网络结构GitHub链接可参考：<https://github.com/pytorch/vision/blob/release/0.7/torchvision/models/resnet.py>。基于ImageNet数据集训练脚本GitHub链接可参考：<https://github.com/pytorch/examples/blob/0.4/imagenet/main.py>

## **支持情况**
### 训练模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50-v1.5  | PyTorch  | MLU370-X8  | FP16/FP32  | Yes  | Not Tested


### 推理模型支持情况
Models  | Framework  | Supported MLU   | Supported Data Precision  
----- | ----- | ----- | ----- | 
ResNet50-v1.5  | PyTorch  | MLU370-S4/X4  | FP16/FP32  |


## 默认参数配置
### **模型训练默认参数配置**
以下为ResNet50模型的默认参数配置：

### Optimizer
模型默认优化器为SGD，以下为相关参数：
* Momentum: 0.9
* Learning Rate: 0.2 for batch size 64
* Learning rate schedule: Linear schedule
* Weight decay: 1e-4
* Label Smoothing: None
* Epoch: 90

### Data Augmentation
模型使用了以下数据增强方法：
* 训练
    * Normolization
    * Crop image to 224*224
    * RandomHorizontalFlip
* 验证
    * Normolization
    * Crop image to 256*256
    * Center crop to 224*224

###  模型推理默认参数配置
* modeldir: 没有指定情况下，默认使用 torchvision 的预训练模型，可选指定训练完成的权重(eg. --modeldir /model/xxxx.pt)
* batch_size: 10,32,64 (batch_size <= 64 in MLU370s4)
* input_data_type：默认使用 float32
* jit&jit_fuse：默认开启
* qint：默认不开启量化
* save_result：默认开启推理结果保存于 result.json 文件


## **依赖项检查**
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.6；
* CNToolKit >=2.8.3;
* CNNL >=1.10.2;
* CNCL >=1.1.1;
* CNLight >=0.12.0;
* CNPyTorch >= 1.3.0;
* 若不具备以上软硬件条件，可前往寒武纪云平台注册并试用@TODO

## **Quick Start Guide**

### 数据集准备
该ResNet50脚本基于ImageNet1K训练，数据集下载：<https://www.image-net.org/>。数据集请放在`/data/pytorch/datasets/imagenet_training`目录下。目录结构为：
```
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── ...
├── train.txt
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── ...
└── val.txt
```


### **环境准备**
#### 基于base docker image安装
##### 1、导入镜像
```
##下载Cambricon PyTorch docker镜像
docker load -i xxx.tar.gz
```

##### 2、启动测试容器
```
bash run_docker.sh
```


##### 3、启动虚拟环境并安装依赖

```
source /torch/venv3/bin/activate
pip install -r requirement.txt
```

#### 使用Dockerfile准备环境
#### 1、生成ResNet50-v1.5的Docker镜像：

```
docker build . -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:resnet50v1.5-catch1.2.0-torch1.6-x86_64-ubuntu18.04-run
```

####  2、创建容器

```
docker run -it --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name mlu_resnet50v1.5 yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:resnet50v1.5-catch1.2.0-torch1.6-x86_64-ubuntu18.04-run 
```

##### 3、启动虚拟环境并安装依赖

```
source /torch/venv3/bin/activate
pip install -r requirement.txt
```

### **Run 脚本执行**
```
eg. bash run_scripts/MLU370S4_ResNet50_Fp16_Int8_Jit.sh
```

#### 一键执行训练脚本
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50-v1.5  | PyTorch  | MLU370-X8  | AMP O1  | 1  | bash MLU370X8_ResNet50_AMP_61E_2MLUs.sh
ResNet50-v1.5  | PyTorch  | MLU370-X8  | FP32  | 2  | bash MLU370X8_ResNet50_FP32_90E_4MLUs.sh
ResNet50-v1.5  | PyTorch  | MLU370-X8  | AMP O1  | 8  | bash MLU370X8_ResNet50_AMP_90E_16MLUs.sh


####  一键执行推理脚本
Models  | Framework  | MLU   | Data Precision  |Run
----- | ----- | ----- | ----- | ----- | 
ResNet50-v1.5  | PyTorch  | MLU370-S4  | FP32  | bash MLU370S4_ResNet50_Fp32.sh
ResNet50-v1.5  | PyTorch  | MLU370-S4  | FP16  | bash MLU370S4_ResNet50_Fp16.sh


### **命令行选项运行**

#### **训练 classify_train.py 所有可选参数如下：**

`python classify_train.py  -h`

```
usage: classify_train.py [-h] [-p N] [-m DIR] [--data DIR] [-j N] [--epochs N]
                   [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--wd W]
                   [--resume_multi_device] [--resume PATH] [-e]
                   [--world-size WORLD_SIZE] [--rank RANK]
                   [--dist-url DIST_URL] [--dist-backend DIST_BACKEND]
                   [--seed SEED] [--save_ckp] [--iters N] [--device DEVICE]
                   [--device_id DEVICE_ID] [--pretrained]
                   [--multiprocessing-distributed] [--ckpdir DIR]
                   [--logdir DIR] [--dummy_test] [--pyamp]
                   [--start_eval_at START_EVAL_AT]
                   [--evaluate_every EVALUATE_EVERY]
                   [--quality_threshold QUALITY_THRESHOLD]

PyTorch ImageNet Training

optional arguments:
  -h, --help            show this help message and exit
  -p N, --print-freq N  print frequency (default: 1)
  -m DIR, --modeldir DIR
                        path to dir of models and mlu operators, default is ./
                        and from torchvision
  --data DIR            path to dataset
  -j N, --workers N     number of data loading works (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --resume_multi_device
                        Only when model is saved by gpu distributed, enable
                        this to load model with submodule
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --save_ckp            Enable save checkpoint
  --iters N             iters per epoch
  --device DEVICE       Use cpu gpu or mlu device
  --device_id DEVICE_ID
                        Use specified device for training, useless in
                        multiprocessing distributed training
  --pretrained          Use a pretrained model
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
  --ckpdir DIR          Where to save ckps
  --logdir DIR          Where to save logs
  --dummy_test          use fake data to traing
  --pyamp               use pytorch amp for mixed precision training
  --start_eval_at START_EVAL_AT
                        start evaluation at specified epoch
  --evaluate_every EVALUATE_EVERY, --eval_every EVALUATE_EVERY
                        evaluate at every epochs
  --quality_threshold QUALITY_THRESHOLD
                        target accuracy
```

####  推理 classif_infer.py 所有可选参数如下：

`python classify_infer.py  -h`

```
usage: classify_infer.py [-h] [-m DIR] [--data DIR] [-b N] [--device DEVICE]
                         [--device_id DEVICE_ID] [--jit JIT]
                         [--jit_fuse JIT_FUSE]
                         [--input_data_type INPUT_DATA_TYPE] [--qint QINT]
                         [--quantized_iters QUANTIZED_ITERS] [--iters N]
                         [--dummy_test] [--save_result SAVE_RESULT]

PyTorch ImageNet Infering

optional arguments:
  -h, --help            show this help message and exit
  -m DIR, --modeldir DIR
                        path to dir of models and mlu operators, default is from torchvision
  --data DIR            path to dataset
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --device DEVICE       Use cpu gpu or mlu device
  --device_id DEVICE_ID
                        Use specified device for infering
  --input_data_type INPUT_DATA_TYPE
                        the input data type, float32 or float16, default
                        float32.
  --iters N             iters per epoch
  --dummy_test          use fake data to traing
  --save_result SAVE_RESULT
                        if save result
```


## **结果展示**

### **训练结果**
##### Training accuracy results: MLU370-X8
Models  | Epochs  | Mixed Precision Top1   | FP32 Top1 
----- | ----- | ----- | ----- | 
ResNet50-v1.5  | 61  | 74.32 | N/A
ResNet50-v1.5  | 100  | 74.44 | 76.124

##### Training performance results: MLU370-X8
Models   | MLUs   | Throughput<br>(FP32)  | Throughput<br>(Mixed Precision)  |  FP32 Training Time<br>(100E) | Mixed Precision Training Time<br>(100E)
----- | ----- | ----- | ----- | ----- | -----|
ResNet50-v1.5  | 1  | 360.71  | 712.24  | N/A| N/A
ResNet50-v1.5  | 4  | 1374.23  | 2604.65  | N/A| N/A
ResNet50-v1.5  | 8  | 2735.88  | 5120  | N/A| N/A


###  推理结果
##### Infering accuracy results: MLU370-S4
Models | (FP32) Top1/Top5   | (FP16) Top1/Top5  |
----- | ----- | ----- | 
ResNet50-v1.5  | 87.19/97.03 | 84.69/95.94 |

##### Infering performance results: MLU370-S4
Models | batch_size  | Throughput<br>(FP32)  | Throughput<br>(FP16)  |
----- |  ----- | ----- | ----- |
ResNet50-v1.5 | 10 | 2005.68fps | 4572.94fps |
ResNet50-v1.5 | 32 | 11114.50fps | 10464.43fps |
ResNet50-v1.5 | 64 | 11823.67fps | 11132.01fps | 

