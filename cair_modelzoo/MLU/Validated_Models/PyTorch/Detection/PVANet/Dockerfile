ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=pytorch_imagenet
RUN git clone https://github.com/sanghoon/pytorch_imagenet.git && cd ${FOLDER_NAME} && git checkout e76c2359b3361b6db0f9faf583487b2b1a784808

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace  --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}