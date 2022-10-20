# pytorch-image-model(TIMM)
## 支持情况

Index | Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- | ----- |
2  | InceptionV4  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
4  | MobileNetV3 | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested 
9  | Xception | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
10 | vovnet | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
11 | DPN68  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
12 | HRNet | PyTorch | MLU370-X8, GPU | FP32 | Yes | Not Tested

### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Classification
```
 
然后通过以下命令运行：
e.g. InceptionV4
其余网络索引见上表
 
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_classify.sh 2 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  | export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_classify.sh 2 fp32-mlu-ddp
V100  | PyTorch  | FP32  | 1  | ./test_classify.sh 2 fp32-gpu
V100  | PyTorch  | FP32  | 8  | ./test_classify.sh 2 fp32-gpu-ddp
