CONFIG_FILE=configs/yolo/yolov3_d53_mstrain-416_273e_coco.py
CHECKPOINT_FILE=/workspace/model/private/tensorrt_infer_yolov3/yolov3_mlu_100.pth
OUTPUT_FILE=/workspace/dataset/private/COCO17/models_yolov3/

# Modify Parameters
cd ./mmdetection
python $(dirname "$0")/tools/config_update.py \
--config=$CONFIG_FILE \
--data_path=/workspace/dataset/private/COCO17/ \
--batch_size=2 \
--interval=1
cd ..


cd ./mmdeploy

# pth to onnx
# python ./tools/deploy.py \
#     configs/mmdet/detection/detection_onnxruntime_dynamic.py \
#     ../mmdetection/${CONFIG_FILE} \
#     ${CHECKPOINT_FILE} \
#     ../mmdetection/demo/demo.jpg \
#     --work-dir ${OUTPUT_FILE} \
#     --show \
#     --device cuda:0

# pth to trt int8
python ./tools/deploy.py \
    configs/mmdet/detection/detection_tensorrt-int8_dynamic-64x64-608x608.py \
    ../mmdetection/${CONFIG_FILE}\
    ${CHECKPOINT_FILE} \
    ../mmdetection/demo/demo.jpg \
    --work-dir ${OUTPUT_FILE} \
    --device cuda:0


