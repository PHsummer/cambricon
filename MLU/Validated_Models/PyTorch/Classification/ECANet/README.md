# ECANet(PyTorch)
## 支持情况

Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
ECANet  | PyTorch  | MLU370-X8, GPU  | FP32  | No  | Not Tested
 
### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Classification
```
 
然后通过以下命令运行：
 
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_classify.sh 0 fp32-mlu
V100  | PyTorch  | FP32  | 1  | ./test_classify.sh 0 fp32-gpu