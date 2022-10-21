ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace
 
ARG FOLDER_NAME=mmsegmentation
RUN git clone https://github.com/open-mmlab/mmsegmentation.git && cd ${FOLDER_NAME} && git checkout a68e7c2823f6c9c79eb2fea68ecbf626aebd2b85 
 
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