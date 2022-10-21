# WRN50-v2(PyTorch)
## 支持情况

Models  | Framework  | Supported MLU   | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- |
WRN50-v2  | PyTorch  | MLU370-X8  | FP32  | Yes  | Not Tested

### Run
Device  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | bash test_classify.sh 8 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  |  MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash test_classify.sh 8 fp32-mlu-ddp
GPU  | PyTorch  | FP32  | 1  | bash test_classify.sh 8 fp32-gpu
GPU  | PyTorch  | FP32  | 8  |  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash test_classify.sh 8 fp32-gpu-ddp
