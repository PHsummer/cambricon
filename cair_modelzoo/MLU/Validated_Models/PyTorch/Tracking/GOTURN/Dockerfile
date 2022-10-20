ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=pygoturn
RUN git clone https://github.com/amoudgl/pygoturn && cd ${FOLDER_NAME} && git checkout 1785ca7106fc0aa9e0fd03b440f34454e8a10e78

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}


RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple \
        opencv-python-headless==4.5.5.64; \
    else \
        pip install -i  https://pypi.tuna.tsinghua.edu.cn/simple \
        opencv-python-headless==4.5.5.64; \
    fi