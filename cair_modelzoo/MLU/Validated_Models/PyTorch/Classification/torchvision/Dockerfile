ARG FROM_IMAGE_NAME
ARG MODEL_NAME
ARG DEVICE_TYPE
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /
 
ARG FOLDER_NAME=examples
RUN git clone https://github.com/pytorch/examples.git && cd ${FOLDER_NAME} && git checkout 2bf23f105237e03ee2501f29670fb6a9ca915096  
 
WORKDIR /${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /${FOLDER_NAME}/; \
    fi
WORKDIR /${FOLDER_NAME}${DEVICE_TYPE}
 
 
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install pyyaml; \
    else \
        pip install pyyaml; \
    fi