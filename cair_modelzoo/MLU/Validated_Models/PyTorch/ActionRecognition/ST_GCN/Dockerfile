ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=st-gcn
RUN git clone https://github.com/fendou201398/st-gcn && cd ${FOLDER_NAME} && git checkout 91e4046fe2274ac74d6220998996cdcd955ba715

WORKDIR /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/update.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=fix update.patch
RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi
WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}


RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        cd torchlight; python setup.py install; cd .. && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        scikit-video==1.1.11 \
        pyyaml==6.0 \
        h5py \
        opencv-python-headless==4.5.5.64; \
    else \
        cd torchlight; python setup.py install; cd .. && \
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        scikit-video==1.1.11 \
        pyyaml==6.0 \
        h5py \
        opencv-python-headless==4.5.5.64; \
    fi