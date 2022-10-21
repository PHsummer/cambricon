ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=CenterNet
RUN git clone https://github.com/xingyizhou/CenterNet.git && cd ${FOLDER_NAME} && git checkout 2b7692c377c6686fb35e473dac2de6105eed62c6

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/${FOLDER_NAME}_mlu_update.patch .
COPY ./${MODEL_NAME}/${FOLDER_NAME}_gpu_update.patch .
RUN apt-get update && apt-get -y install libgl1-mesa-glx 
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        git apply --ignore-space-change --ignore-whitespace  --whitespace=fix ${FOLDER_NAME}_mlu_update.patch; \
    else \
        git apply --ignore-space-change --ignore-whitespace  --whitespace=fix ${FOLDER_NAME}_gpu_update.patch; \
    fi
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}/src/

RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install pycocotools && \
        pip install -r ../requirements.txt && \
        cd /workspace/CenterNet_mlu/src/lib/models/networks/DCNv2 && \
        bash ./make.sh; \
    else \ 
        pip install -r ../requirements.txt && \
        pip install pycocotools; \
    fi
