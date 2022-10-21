#  TensorRT Inference
##  ResNet50
### PTH to ONNX
```
python onnx_export.py
```

### ONNX to int8 quant
```
bash quant.sh
```

### Validation
####  for pth
```
# python validation.py \
# -a resnet50 \
# --mode pth \
# --batch-size 8 \
# --data /workspace/dataset/private/imagenet_training \
# --resume /workspace/algorithm/models/pytorch/resnet50/resnet50_gpu_98.pth 
```

#### for onnx
```
python validation.py \
-a resnet50 \
--mode onnx \
--batch-size 8 \
--data /workspace/dataset/private/imagenet_training \
--resume /workspace/model/private/tensorrt_infer/onnx/resnet50/resnet50_gpu_98_op11.onnx
```

#### for int8 quant
```
# python validation.py \
# -a resnet50 \
# --mode trt \
# --batch-size 8 \
# --data /workspace/dataset/private/imagenet_training \
# --resume ./models/trt_int8/resnet50/bs8_test/resnet50_gpu_op11_bs8.int8.engine
```