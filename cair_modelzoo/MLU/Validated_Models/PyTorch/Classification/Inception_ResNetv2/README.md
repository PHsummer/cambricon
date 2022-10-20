# Inception_ResNetv2(PyTorch)
## 支持情况

Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
Inception_ResNetv2  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
 
### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Classification
```
 
然后通过以下命令运行：
 
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- | 
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_classify.sh 1 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  | export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_classify.sh 1 fp32-mlu-ddp
V100  | PyTorch  | FP32  | 1  | ./test_classify.sh 1 fp32-gpu
V100  | PyTorch  | FP32  | 8  | export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_classify.sh 1 fp32-gpu-ddp
