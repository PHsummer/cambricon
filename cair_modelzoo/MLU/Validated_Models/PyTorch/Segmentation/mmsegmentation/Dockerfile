ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace
 
ARG FOLDER_NAME=mmsegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git && cd ${FOLDER_NAME} && git checkout 4d0eb367e9136c0000a5ee9ee45de1db3a557418 
 
WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
 
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip uninstall -y mmcv mmcv-full && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        mmcls \
        mmpycocotools \
        prettytable \
        openmim==0.2.0 && \
        mim install mmcv-full==1.5.0; \
    else \
        pip uninstall -y mmcv mmcv-full && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        mmcls \
        mmpycocotools \
        prettytable \
        openmim==0.2.0 && \
        mim install mmcv-full==1.5.0; \
    fi