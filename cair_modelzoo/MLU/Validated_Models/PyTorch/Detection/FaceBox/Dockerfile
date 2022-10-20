ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=FaceBoxes.PyTorch
RUN git clone https://github.com/zisianw/FaceBoxes.PyTorch.git && cd ${FOLDER_NAME} && git checkout 9bc5811fe8c409a50c9f23c6a770674d609a2c3a

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace  --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}

RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install matplotlib && pip install opencv-python==4.5.5.64; \
    else \
        apt-get update && \
        apt-get -y install libgl1-mesa-glx && \
        pip install matplotlib && pip install opencv-python==4.5.5.64; \
    fi



