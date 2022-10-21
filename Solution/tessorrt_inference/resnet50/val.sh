pth val
python validation.py \
-a resnet50 \
--mode pth \
--batch-size 8 \
--data /workspace/dataset/private/imagenet_training \
--resume /workspace/algorithm/models/pytorch/resnet50/resnet50_gpu_98.pth 
2>&1 | tee resnet50_gpu_fp32_8.log

# onnx val
# python validation.py \
# -a resnet50 \
# --mode onnx \
# --batch-size 8 \
# --data /workspace/dataset/private/imagenet_training \
# --resume /workspace/model/private/tensorrt_infer/onnx/resnet50/resnet50_gpu_98_op11.onnx
# 2>&1 | tee ./log/bs1_test/resnet50_mlu_op10_onnx_1.log

# TensorRT val
# python validation.py \
# -a resnet50 \
# --mode trt \
# --batch-size 8 \
# --data /workspace/dataset/private/imagenet_training \
# --resume ./models/trt_int8/resnet50/bs8_test/resnet50_gpu_op11_bs8.int8.engine
# 2>&1 | tee resnet50_gpu_fp32_8.log