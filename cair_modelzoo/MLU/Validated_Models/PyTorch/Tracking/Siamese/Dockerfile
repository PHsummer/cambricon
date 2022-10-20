ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=siamese-pytorch
RUN git clone https://github.com/fangpin/siamese-pytorch && cd ${FOLDER_NAME} && git checkout 5543f1e844964b116dc9d347a5eb164c6a7afe6d

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}


RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install matplotlib==3.3.4 && \
        pip install python-gflags==3.1.2; \
    else \
        pip install matplotlib==3.3.4 && \
        pip install python-gflags==3.1.2; \
    fi

RUN git clone https://github.com/brendenlake/omniglot.git
RUN cd omniglot/python && unzip images_evaluation.zip
RUN cd omniglot/python && unzip images_background.zip
