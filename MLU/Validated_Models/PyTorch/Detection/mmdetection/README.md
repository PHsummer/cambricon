# MMDetection(PyTorch)
## 支持情况

Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
MobileNetV2_SSDlite  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested

 
### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Detection
```
 
然后通过以下命令运行：

e.g. MobileNetV2_SSDlite  
  
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_detection.sh 10 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  | export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_detection.sh 10 fp32-mlu-ddp
V100  | PyTorch  | FP32  | 1  | ./test_detection.sh 10 fp32-gpu
V100  | PyTorch  | FP32  | 8  | export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_detection.sh 10 fp32-gpu-ddp
