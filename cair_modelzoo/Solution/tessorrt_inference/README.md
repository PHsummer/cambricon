# ResNet50 TRT推理
## 1 模型转换
```
cd resnet50
```
### 1.1 PTH to ONNX
修改参数
- model_src：PTH模型路径
- model_dst：输出的ONNX模型路径，含完整的ONNX模型名称及".onnx"后缀
- shape：输入数据shape
- opset：opset version
  
运行
```
python onnx_export.py \
    --model_src $PTH_PATH \
    --model_dst $ONNX_SAVE \
    --shape (3,224,224) \
    --opset 10
```
### 1.2 ONNX to trt int8
修改参数
- CALIBRATION_DATA：用于量化的后训练数据路径
- MAX_CALIBRATION_SIZE：后训练的数据量，可自由调整
- CACHE_FILENAME：中间过程文件存储路径
- PREPROCESS_FUNC：前处理函数名称，位于processing.py内，可自写其他网络前处理函数
- ONNX_MODEL：上述转出的onnx路径
- OUTPUT：输出的int8模型路径，含完整的模型名称及".engine"后缀 
  
运行  
```
bash quant.sh
```
## 2 测试
说明
- 脚本内涵盖pth、onnx及trt三种模型的测试命令行，可根据测试模型类型，注释或取消注释val.sh内对应的命令行；  
- 不同batch size及不同模型需分多次测试；  
  
修改参数
- mode：可选择的推理类型，pth、onnx或trt
- batch-size：测试的batch size
- data：测试数据路径，通常为训练数据路径，自动读取测试数据集
- resume：测试模型路径
  
运行
```
bash val.sh
```

# YOLOv3 TRT推理

```
cd yolov3
```
## 1 环境搭建
### 1.1 说明
所需docker image环境为 
```
nvcr.io/nvidia/pytorch:21.09-py3
```
setup.sh脚本包含完整环境的部署过程，无需任何修改，在进入docker后直接运行：
```
bash setup.sh
```
pth测试由mmdet完成，trt int8模型由mmdeploy完成；
  
### 1.2 可能的报错：
```
KeyError: 'Cannot get key by value "Backend.TENSORRT" of <enum \'Backend\'>'
``` 
解决方法：  
修改文件 /opt/conda/lib/python3.8/site-packages/mmdeploy/utils/constants.py  
Line 14  
```
if k.value == value:
```
修改为
```
if k.value == value or k == value:
```
  
## 2 模型转换
可直接转换ONNX模型及trt量化模型  
  
修改参数
- CONFIG_FILE: mmdet中的模型配置文件路径，与训练中的配置文件一致
- CHECKPOINT_FILE: PTH路径
- OUTPUT_FILE：onnx模型及int8模型engine文件的输出路径，不含文件名及后缀
- data_path：训练数据地址。代码自动读取其中的测试数据集
  
运行
```
bash model_convert.sh
```
  
## 3 测试
说明
- 不同类型模型可同时测试，不同batch size需分多次测试。  
  
修改参数
- BATCH_SIZE：测试的batch size
- OPSET_VERSION：暂不支持opset_version=10
- CONFIG_FILE：mmdet中的模型配置文件路径，与训练中的配置文件一致
- CONFIG_QUANT：mmdeploy中的量化配置文件
- CHECKPOINT_FILE_GPU：GPU的pth模型路径
- CHECKPOINT_FILE_MLU：MLU的pth模型路径
- TRT_FILE_GPU：GPU的int8量化的engine模型
- TRT_FILE_MLU：MLU的int8量化的engine模型
- data_path：训练数据地址。代码自动读取其中的测试数据集
- 对应的log文件路径  
  
运行
```
bash model_eval.sh
```
