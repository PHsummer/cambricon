ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=kg-bert
RUN git clone https://github.com/yao8839836/kg-bert && cd kg-bert && git checkout 0b8f625108efcb9b43603b6840f2548ff3d3c973

WORKDIR  /workspace/${FOLDER_NAME}/
COPY ./${MODEL_NAME}/kg_bert_diff.patch .
RUN patch -p1 < kg_bert_diff.patch

RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi

WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}

RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && \
        pip install pytorch-pretrained-bert==0.6.2 tqdm boto3 requests wikipedia; \
    else \
        pip install pytorch-pretrained-bert==0.6.2 tqdm boto3 requests wikipedia; \
    fi