# ShuffleNet_1x_g3(PyTorch)

## 支持情况

| Models      | Framework | Supported MLU | Supported Data Precision | Multi-MLUs | Multi-Nodes |
| ----------- | --------- | ------------- | ------------------------ | ---------- | ----------- |
| ShuffleNet_1x_g3 | PyTorch   | MLU370-X8     | FP32                     | No         | Not Tested  |

### Run

先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Classification
```

然后通过以下命令运行：
 
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_classify.sh 12 fp32-mlu
V100  | PyTorch  | FP32  | 1  | ./test_classify.sh 12 fp32-gpu
