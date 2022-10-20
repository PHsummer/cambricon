# MMSegmentation(PyTorch)
## 支持情况

Index  |  Models  | Framework  | Supported Device  | Supported Data Precision  | Multi-MLUs  | Multi-Nodes
----- | ----- | ----- | ----- | ----- | ----- | ----- |
0 | ERFNet  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
1 | ICNet  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
2 | SegFormer  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
3 | Twins  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
4 | FCN  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
5 | PSPNet  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
6 | BiSeNet  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
7 | FCN_R50  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested
8 | DeeplabV3_R50  | PyTorch  | MLU370-X8, GPU  | FP32  | Yes  | Not Tested

 

### Run
先进入分类目录下：
 
```
cd ./cair_modelzoo/MLU/Validated_Models/PyTorch/Segmentation
```
 
然后通过以下命令运行：

e.g. ERFNet（index: 0）  
其余网络索引见上表  
  
MLU  | Framework  |  Data Precision  | Cards  | Run
----- | ----- | ----- | ----- | ----- |
MLU370-X8  | PyTorch  | FP32  | 1  | ./test_segmentation.sh 0 fp32-mlu
MLU370-X8  | PyTorch  | FP32  | 8  | export MLU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_segmentation.sh 0 fp32-mlu-ddp
V100  | PyTorch  | FP32  | 1  | ./test_segmentation.sh 0 fp32-gpu
V100  | PyTorch  | FP32  | 8  | export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && ./test_segmentation.sh 0 fp32-gpu-ddp
