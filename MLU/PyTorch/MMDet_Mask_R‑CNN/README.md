# MMDet_Mask_R-CNN(PyTorch)
## 支持情况

Models  | Framework  | MLU | Data Precision  | Device Number | Epochs  | Accuracy
----- | ----- | ----- | ----- | ----- | ----- | ----- |
MMDet_Mask_R-CNN  | PyTorch  | MLU370-X8  | FP32  | 1  | 12  | 'bbox_mAP': 0.4, 'segm_mAP': 0.361

## Quick Start Guide
### 数据集
数据集路径为`/data/pytorch/datasets/COCO2017`。
### Docker镜像
生成MMDet_Mask_R-CNN的Docker镜像：`docker build -t yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:mmdet_maskrcnn-catch1.2.1-torch1.6-x86_64-ubuntu18.04 .`
### Run
MLU370-X8单卡FP32测试精度指令：
`bash pt-mmdet_maskrcnn-one-X8-from-scratch.sh`