# Path to dataset to use for calibration. 
#   **Not necessary if you already have a calibration cache from a previous run.
CALIBRATION_DATA="/workspace/dataset/private/imagenet_training/posttrain"

# Truncate calibration images to a random sample of this amount if more are found.
#   **Not necessary if you already have a calibration cache from a previous run.
MAX_CALIBRATION_SIZE=50000

# Calibration cache to be used instead of calibration data if it already exists,
# or the cache will be created from the calibration data if it doesn't exist.
CACHE_FILENAME="/workspace/algorithm/models/trt_int8/resnet50/bs8_test/resnet50_mlu_op10_bs8.cache"

# Any function name defined in `processing.py`
PREPROCESS_FUNC="preprocess_imagenet"

# Path to ONNX model
ONNX_MODEL="/workspace/model/private/tensorrt_infer/onnx/resnet50/resnet50_mlu_100_op10.onnx"

# Path to write TensorRT engine to
OUTPUT="/workspace/algorithm/models/trt_int8/resnet50/bs8_test/resnet50_mlu_op10_bs8.int8.engine"

# Creates an int8 engine from your ONNX model, creating ${CACHE_FILENAME} based
# on your ${CALIBRATION_DATA}, unless ${CACHE_FILENAME} already exists, then
# it will use simply use that instead.
python utils/onnx_to_tensorrt.py --fp16 --int8 -v \
        -b 8 \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT} 

        