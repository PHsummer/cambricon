# YOLOv3 ONNX export
BATCH_SIZE=1
OPSET_VERSION=11
CONFIG_FILE=configs/yolo/yolov3_d53_mstrain-416_273e_coco.py
CONFIG_QUANT=configs/mmdet/detection/detection_tensorrt-int8_dynamic-64x64-608x608.py
CHECKPOINT_FILE_GPU=/workspace/model/private/tensorrt_infer_yolov3/yolov3_gpu_100.pth
CHECKPOINT_FILE_MLU=/workspace/model/private/tensorrt_infer_yolov3/yolov3_mlu_100.pth
TRT_FILE_GPU=/workspace/dataset/private/COCO17/models_yolov3/yolov3_gpu_op${OPSET_VERSION}_int8.engine
TRT_FILE_MLU=/workspace/dataset/private/COCO17/models_yolov3/yolov3_mlu_op${OPSET_VERSION}_int8.engine

###########
# PTH Test
###########
cd ./mmdetection
python $(dirname "$0")/tools/config_update.py \
--config=$CONFIG_FILE \
--data_path=/workspace/dataset/private/COCO17/ \
--batch_size=$BATCH_SIZE \
--interval=1

python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE_GPU} \
    --eval bbox \
    --batch_size ${BATCH_SIZE} \
2>&1 | tee yolov3_gpu_pth_bs${BATCH_SIZE}.log

python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE_MLU} \
    --eval bbox \
    --batch_size ${BATCH_SIZE} \
2>&1 | tee yolov3_mlu_pth_bs${BATCH_SIZE}.log

cd ..


###########
# TRT Test
###########
cd ./mmdeploy

python tools/test.py \
    ${CONFIG_QUANT} \
    ../mmdetection/${CONFIG_FILE} \
    --model ${TRT_FILE_GPU} \
    --batch_size $BATCH_SIZE \
    --metrics bbox \
    --device cuda:0 \
2>&1 | tee /workspace/dataset/private/COCO17/models_yolov3/log/yolov3_gpu_op${OPSET_VERSION}_bs${BATCH_SIZE}_int8.log

python tools/test.py \
    ${CONFIG_QUANT} \
    ../mmdetection/${CONFIG_FILE} \
    --model ${TRT_FILE_MLU} \
    --batch_size $BATCH_SIZE \
    --metrics bbox \
    --device cuda:0 \
2>&1 | tee /workspace/dataset/private/COCO17/models_yolov3/log/yolov3_mlu_op${OPSET_VERSION}_bs${BATCH_SIZE}_int8.log

cd ..
