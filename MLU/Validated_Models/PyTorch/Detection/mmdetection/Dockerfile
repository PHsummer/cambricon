ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace
 
ARG FOLDER_NAME=mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git && cd ${FOLDER_NAME} && git checkout ca11860f4f3c3ca2ce8340e2686eeaec05b29111 
 
WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
 
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip uninstall -y mmcv mmcv-full && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        pycocotools \
        openmim==0.2.0 && \
        mim install mmcv-full==1.6.0; \
    else \
        pip uninstall -y mmcv mmcv-full && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        pycocotools \
        terminaltables \
        openmim==0.2.0 && \
        mim install mmcv-full==1.6.0; \
    fi