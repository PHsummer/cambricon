# MMDetection(PyTorch)
## 支持情况

Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
PVANet  | PyTorch  | MLU370-X8, GPU  | FP32  | NO  | Not Tested

 
### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Detection
```
 
然后通过以下命令运行：

e.g. PVANet  
  
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_detection.sh 3 fp32-mlu
V100  | PyTorch  | FP32  | 1  | ./test_detection.sh 3 fp32-gpu