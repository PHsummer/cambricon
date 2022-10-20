# CAIR ModelZoo for Cambricon MLU Accelerators
## Introduction
**cair_modelzoo**提供可以在寒武纪MLU设备上运行的业界公认benchmark的模型集。
## Computer Vision

Models  | Framework  | Device | Precision  | Multi-MLU  | Multi-Node
----- | ----- | ----- | ----- | ----- | ----- |
ResNet50  | PyTorch  | MLU370-X8  | Native Torch AMP  | Yes  | -
MMDet_Mask_R-CNN  | PyTorch  | MLU370-X8  | FP32  | Yes  | -
SSD-VGG16 | PyTorch  | MLU370-X8  | AMP  | Yes  | -
Cifar100_ResNet50  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | Not Tested
EfficientNetv2_rw_t  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | Yes
Yolov5m_v6.0  | PyTorch  | MLU370-X8  | FP32/AMP  | Yes  | Yes
ResNet34  | PyTorch  | MLU290  | FP32  | Yes  | Not Tested
VGG19  | PyTorch  | MLU290  | FP32  | Yes  | Not Tested
VGG16-BN | PyTorch  | MLU290  | FP32  | Yes  | Not Tested
ShuffleNet-V2 | PyTorch  | MLU290  | FP32  | Yes  | Not Tested
ResNetV1D-50 | PyTorch  | MLU290  | FP32  | Yes  | Not Tested
SE-ResNet-50 | PyTorch  | MLU290  | FP32  | Yes  | Not Tested

## NLP

Models  | Framework  | Device  | Precision  | Multi-MLU  | Multi-Node
----- | ----- | ----- | ----- | ----- | ----- |
BERT-Large  | PyTorch  | MLU370-X8  | Native Torch AMP  | Yes  | -
GLM-base | DeepSpeed | MLU370-X8  | Native Torch AMP  | Yes  | -

## Recommendation
Models  | Framework  | Device  | Precision  | Multi-MLU  | Multi-Node
----- | ----- | ----- | ----- | ----- | ----- |
DLRM | PyTorch | MLU370-X8 | Native Torch AMP  | Yes  | -

## Text To Speech
Models  | Framework  | Device  | Precision  | Multi-MLU  | Multi-Node
----- | ----- | ----- | ----- | ----- | ----- |
Tacotron2 | PyTorch  | MLU370-X8  | Native Torch AMP  | Yes  | -

## Automated speech recognition
Models  | Framework  | Device  | Precision  | Multi-MLU  | Multi-Node
----- | ----- | ----- | ----- | ----- | ----- |
Conformer | Pytorch | MLU370-X8 | FP32/AMP | Yes | Not tested
