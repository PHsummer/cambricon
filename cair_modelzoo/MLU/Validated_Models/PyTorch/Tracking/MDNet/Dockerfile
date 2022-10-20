ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=py-MDNet
RUN git clone https://github.com/hyeonseobnam/py-MDNet && cd ${FOLDER_NAME} && git checkout 680fa4d58c427b8b647fa59dd9cd61a9fb7061f6

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}


RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        pyyaml==6.0 \
        opencv-python-headless==4.5.5.64 \
        xmltodict==0.13.0; \
    else \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        pyyaml==6.0 \
        opencv-python-headless==4.5.5.64 \
        xmltodict==0.13.0; \
    fi


RUN cd models && wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-m.mat