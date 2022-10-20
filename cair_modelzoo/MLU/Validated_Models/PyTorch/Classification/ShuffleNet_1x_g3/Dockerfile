ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=ShuffleNet-Series
RUN git clone https://github.com/megvii-model/ShuffleNet-Series.git && cd ${FOLDER_NAME} && git checkout aa91feb71b01f28d0b8da3533d20a3edb11b1810  

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}