# timm 0.5.4 -- efficientnetv2_rw_t(PyTorch)
## 模型概述
`EfficientNetv2_rw_t`网络是基于 [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/pdf/2104.00298.pdf)的tiny版本，出自rwightman创建的timm库。

本仓库为[timm(v0.5.4)](https://github.com/rwightman/pytorch-image-models/tree/v0.5.4)的MLU实现。

## 支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-GPUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- 
EfficientNetv2_rw_t  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | Yes 

## Quick Start Guide
### 数据集
该`efficientnetv2_rw_t`脚本基于ImageNet1K训练，数据集下载：<https://www.image-net.org/>。数据集请放在`/data/pytorch/datasets/imagenet_training`目录下。目录结构为：
```bash
├── train
│   ├── n01440764
│   ├── n01443537
│   ├── ...
├── train.txt
├── val
│   ├── n01440764
│   ├── n01443537
│   ├── ...
└── val.txt
```

### Set up
#### 基于base docker images安装
##### 1、导入镜像
```bash
##下载Cambriocn PyTorch docker镜像
docker load -i xxx.tar.gz
```
##### 2、启动测试容器
```bash
docker run -it --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name mlu_efficientnet xxx
```
##### 3、启动虚拟环境并安装依赖
```bash
source /torch/venv3/pytorch/bin/activate
pip install -r requirements.txt
```

#### 使用Dockerfile准备环境
##### 1、生成EfficientNetv2-rw-t的Docker镜像：

```bash
docker build -t efficientnet_pytorch_1.9.0_docker:final -f Dockerfile .
```

#####  2、创建容器

```bash
docker run -it --ipc=host -v /data:/data  --device /dev/cambricon_ctl --privileged --name mlu_efficientnet efficientnet_pytorch_1.9.0_docker:final
```

##### 3、启动虚拟环境并安装依赖

```bash
source /torch/venv3/pytorch/bin/activate
pip install -r requirements.txt
```


## 默认参数配置
以下为`EfficientNet-rw-t`模型的默认参数配置：

### Optimizer
模型默认优化器为RMSProp，以下为相关参数：

* decay: 0.9
* momentum: 0.9
* Learning Rate: 0 to 0.256
* Weight decay: 1e-5
* Epoch: 520
* EMA: decay rate 0.9999

模型使用了以下数据增强方法：

* Color Jitter
* Mixup
* RandAugment
* Cutmix
* Random Erasing

### 依赖
* Linux常见操作系统版本(如Ubuntu16.04，Ubuntu18.04，CentOS7.x等)，安装docker(>=v18.00.0)应用程序；
* 服务器装配好寒武纪计算版本MLU370-X8;
* Cambricon Driver >=v4.20.11；
* CNToolKit >=2.8.5-1;
* CNNL >=1.10.5-topkfixnan;
* CNCL >=1.1.2-1;
* CNPyTorch >= 1.3.2;

## 命令行选项

train.py所有可选参数如下：

`python train.py  -h`

```bash
usage: train.py [-h] [--dataset NAME] [--train-split NAME] [--val-split NAME]
                [--dataset-download] [--class-map FILENAME] [--model MODEL]
                [--pretrained] [--initial-checkpoint PATH] [--resume PATH]
                [--no-resume-opt] [--num-classes N] [--gp POOL] [--img-size N]
                [--input-size N N N N N N N N N] [--crop-pct N]
                [--mean MEAN [MEAN ...]] [--std STD [STD ...]]
                [--interpolation NAME] [-b N] [-vb N] [--opt OPTIMIZER]
                [--opt-eps EPSILON] [--opt-betas BETA [BETA ...]]
                [--momentum M] [--weight-decay WEIGHT_DECAY]
                [--clip-grad NORM] [--clip-mode CLIP_MODE] [--sched SCHEDULER]
                [--lr LR] [--lr-noise pct, pct [pct, pct ...]]
                [--lr-noise-pct PERCENT] [--lr-noise-std STDDEV]
                [--lr-cycle-mul MULT] [--lr-cycle-decay MULT]
                [--lr-cycle-limit N] [--lr-k-decay LR_K_DECAY]
                [--warmup-lr LR] [--min-lr LR] [--epochs N]
                [--epoch-repeats N] [--start-epoch N] [--decay-epochs N]
                [--warmup-epochs N] [--cooldown-epochs N]
                [--patience-epochs N] [--decay-rate RATE] [--no-aug]
                [--scale PCT [PCT ...]] [--ratio RATIO [RATIO ...]]
                [--hflip HFLIP] [--vflip VFLIP] [--color-jitter PCT]
                [--aa NAME] [--aug-repeats AUG_REPEATS]
                [--aug-splits AUG_SPLITS] [--jsd-loss] [--bce-loss]
                [--bce-target-thresh BCE_TARGET_THRESH] [--reprob PCT]
                [--remode REMODE] [--recount RECOUNT] [--resplit]
                [--mixup MIXUP] [--cutmix CUTMIX]
                [--cutmix-minmax CUTMIX_MINMAX [CUTMIX_MINMAX ...]]
                [--mixup-prob MIXUP_PROB]
                [--mixup-switch-prob MIXUP_SWITCH_PROB]
                [--mixup-mode MIXUP_MODE] [--mixup-off-epoch N]
                [--smoothing SMOOTHING]
                [--train-interpolation TRAIN_INTERPOLATION] [--drop PCT]
                [--drop-connect PCT] [--drop-path PCT] [--drop-block PCT]
                [--bn-momentum BN_MOMENTUM] [--bn-eps BN_EPS] [--sync-bn]
                [--dist-bn DIST_BN] [--split-bn] [--model-ema]
                [--model-ema-force-cpu] [--model-ema-decay MODEL_EMA_DECAY]
                [--seed S] [--worker-seeding WORKER_SEEDING]
                [--log-interval N] [--recovery-interval N]
                [--checkpoint-hist N] [-j N] [--save-images] [--amp]
                [--apex-amp] [--native-amp] [--no-ddp-bb] [--channels-last]
                [--pin-mem] [--no-prefetcher] [--output PATH]
                [--experiment NAME] [--eval-metric EVAL_METRIC] [--tta N]
                [--local_rank LOCAL_RANK] [--use-multi-epochs-loader]
                [--torchscript] [--log-wandb]
                DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset NAME, -d NAME
                        dataset type (default: ImageFolder/ImageTar if empty)
  --train-split NAME    dataset train split (default: train)
  --val-split NAME      dataset validation split (default: validation)
  --dataset-download    Allow download of dataset for torch/ and tfds/
                        datasets that support it.
  --class-map FILENAME  path to class to idx mapping file (default: "")
  --model MODEL         Name of model to train (default: "resnet50"
  --pretrained          Start with pretrained version of specified network (if
                        avail)
  --initial-checkpoint PATH
                        Initialize model from this checkpoint (default: none)
  --resume PATH         Resume full model and optimizer state from checkpoint
                        (default: none)
  --no-resume-opt       prevent resume of optimizer state when resuming model
  --num-classes N       number of label classes (Model default if None)
  --gp POOL             Global pool type, one of (fast, avg, max, avgmax,
                        avgmaxc). Model default if None.
  --img-size N          Image patch size (default: None => model default)
  --input-size N N N N N N N N N
                        Input all image dimensions (d h w, e.g. --input-size 3
                        224 224), uses model default if empty
  --crop-pct N          Input image center crop percent (for validation only)
  --mean MEAN [MEAN ...]
                        Override mean pixel value of dataset
  --std STD [STD ...]   Override std deviation of of dataset
  --interpolation NAME  Image resize interpolation type (overrides model)
  -b N, --batch-size N  input batch size for training (default: 128)
  -vb N, --validation-batch-size N
                        validation batch size override (default: None)
  --opt OPTIMIZER       Optimizer (default: "sgd"
  --opt-eps EPSILON     Optimizer Epsilon (default: None, use opt default)
  --opt-betas BETA [BETA ...]
                        Optimizer Betas (default: None, use opt default)
  --momentum M          Optimizer momentum (default: 0.9)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 2e-5)
  --clip-grad NORM      Clip gradient norm (default: None, no clipping)
  --clip-mode CLIP_MODE
                        Gradient clipping mode. One of ("norm", "value",
                        "agc")
  --sched SCHEDULER     LR scheduler (default: "step"
  --lr LR               learning rate (default: 0.05)
  --lr-noise pct, pct [pct, pct ...]
                        learning rate noise on/off epoch percentages
  --lr-noise-pct PERCENT
                        learning rate noise limit percent (default: 0.67)
  --lr-noise-std STDDEV
                        learning rate noise std-dev (default: 1.0)
  --lr-cycle-mul MULT   learning rate cycle len multiplier (default: 1.0)
  --lr-cycle-decay MULT
                        amount to decay each learning rate cycle (default:
                        0.5)
  --lr-cycle-limit N    learning rate cycle limit, cycles enabled if > 1
  --lr-k-decay LR_K_DECAY
                        learning rate k-decay for cosine/poly (default: 1.0)
  --warmup-lr LR        warmup learning rate (default: 0.0001)
  --min-lr LR           lower lr bound for cyclic schedulers that hit 0 (1e-5)
  --epochs N            number of epochs to train (default: 300)
  --epoch-repeats N     epoch repeat multiplier (number of times to repeat
                        dataset epoch per train epoch).
  --start-epoch N       manual epoch number (useful on restarts)
  --decay-epochs N      epoch interval to decay LR
  --warmup-epochs N     epochs to warmup LR, if scheduler supports
  --cooldown-epochs N   epochs to cooldown LR at min_lr, after cyclic schedule
                        ends
  --patience-epochs N   patience epochs for Plateau LR scheduler (default: 10
  --decay-rate RATE, --dr RATE
                        LR decay rate (default: 0.1)
  --no-aug              Disable all training augmentation, override other
                        train aug args
  --scale PCT [PCT ...]
                        Random resize scale (default: 0.08 1.0)
  --ratio RATIO [RATIO ...]
                        Random resize aspect ratio (default: 0.75 1.33)
  --hflip HFLIP         Horizontal flip training aug probability
  --vflip VFLIP         Vertical flip training aug probability
  --color-jitter PCT    Color jitter factor (default: 0.4)
  --aa NAME             Use AutoAugment policy. "v0" or "original". (default:
                        None)
  --aug-repeats AUG_REPEATS
                        Number of augmentation repetitions (distributed
                        training only) (default: 0)
  --aug-splits AUG_SPLITS
                        Number of augmentation splits (default: 0, valid: 0 or
                        >=2)
  --jsd-loss            Enable Jensen-Shannon Divergence + CE loss. Use with
                        `--aug-splits`.
  --bce-loss            Enable BCE loss w/ Mixup/CutMix use.
  --bce-target-thresh BCE_TARGET_THRESH
                        Threshold for binarizing softened BCE targets
                        (default: None, disabled)
  --reprob PCT          Random erase prob (default: 0.)
  --remode REMODE       Random erase mode (default: "pixel")
  --recount RECOUNT     Random erase count (default: 1)
  --resplit             Do not random erase first (clean) augmentation split
  --mixup MIXUP         mixup alpha, mixup enabled if > 0. (default: 0.)
  --cutmix CUTMIX       cutmix alpha, cutmix enabled if > 0. (default: 0.)
  --cutmix-minmax CUTMIX_MINMAX [CUTMIX_MINMAX ...]
                        cutmix min/max ratio, overrides alpha and enables
                        cutmix if set (default: None)
  --mixup-prob MIXUP_PROB
                        Probability of performing mixup or cutmix when
                        either/both is enabled
  --mixup-switch-prob MIXUP_SWITCH_PROB
                        Probability of switching to cutmix when both mixup and
                        cutmix enabled
  --mixup-mode MIXUP_MODE
                        How to apply mixup/cutmix params. Per "batch", "pair",
                        or "elem"
  --mixup-off-epoch N   Turn off mixup after this epoch, disabled if 0
                        (default: 0)
  --smoothing SMOOTHING
                        Label smoothing (default: 0.1)
  --train-interpolation TRAIN_INTERPOLATION
                        Training interpolation (random, bilinear, bicubic
                        default: "random")
  --drop PCT            Dropout rate (default: 0.)
  --drop-connect PCT    Drop connect rate, DEPRECATED, use drop-path (default:
                        None)
  --drop-path PCT       Drop path rate (default: None)
  --drop-block PCT      Drop block rate (default: None)
  --bn-momentum BN_MOMENTUM
                        BatchNorm momentum override (if not None)
  --bn-eps BN_EPS       BatchNorm epsilon override (if not None)
  --sync-bn             Enable NVIDIA Apex or Torch synchronized BatchNorm.
  --dist-bn DIST_BN     Distribute BatchNorm stats between nodes after each
                        epoch ("broadcast", "reduce", or "")
  --split-bn            Enable separate BN layers per augmentation split.
  --model-ema           Enable tracking moving average of model weights
  --model-ema-force-cpu
                        Force ema to be tracked on CPU, rank=0 node only.
                        Disables EMA validation.
  --model-ema-decay MODEL_EMA_DECAY
                        decay factor for model weights moving average
                        (default: 0.9998)
  --seed S              random seed (default: 42)
  --worker-seeding WORKER_SEEDING
                        worker seed mode (default: all)
  --log-interval N      how many batches to wait before logging training
                        status
  --recovery-interval N
                        how many batches to wait before writing recovery
                        checkpoint
  --checkpoint-hist N   number of checkpoints to keep (default: 10)
  -j N, --workers N     how many training processes to use (default: 4)
  --save-images         save images of input bathes every log interval for
                        debugging
  --amp                 use NVIDIA Apex AMP or Native AMP for mixed precision
                        training
  --apex-amp            Use NVIDIA Apex AMP mixed precision
  --native-amp          Use Native Torch AMP mixed precision
  --no-ddp-bb           Force broadcast buffers for native DDP to off.
  --channels-last       Use channels_last memory layout
  --pin-mem             Pin CPU memory in DataLoader for more efficient
                        (sometimes) transfer to GPU.
  --no-prefetcher       disable fast prefetcher
  --output PATH         path to output folder (default: none, current dir)
  --experiment NAME     name of train experiment, name of sub-folder for
                        output
  --eval-metric EVAL_METRIC
                        Best metric (default: "top1"
  --tta N               Test/inference time augmentation (oversampling)
                        factor. 0=None (default: 0)
  --local_rank LOCAL_RANK
  --use-multi-epochs-loader
                        use the multi-epochs-loader to save time at the
                        beginning of every epoch
  --torchscript         convert model torchscript for inference
  --log-wandb           log training and validation metrics to wandb
```

## Advanced
### Run
Models  | Framework  | MLU   | Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | ----- 
EfficientNetv2-rw-t | PyTorch  | MLU370-X8  | AMP | 8  | bash MLU370X8_EfficientNetv2-rw-t_AMP_520E_16MLUs.sh 


### 训练结果
#### Training accuracy results: MLU370-X8
Models  | Epochs  | ACC@top1 | ACC@top5
----- | ----- | ----- | -----
EfficientNetv2-rw-t  | 517    | 82.54 | 96.25

#### Training performance results: MLU370-X8

Models   | MLUs   | batch size  | Throughput(Mixed Precision)  
----- | ----- | ----- | ----- 
EfficientNetv2-rw-t  | 8  | 164*16     | 2876.73 fps
