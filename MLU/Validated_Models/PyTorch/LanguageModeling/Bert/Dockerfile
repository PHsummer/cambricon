ARG FROM_IMAGE_NAME
FROM ${FROM_IMAGE_NAME}
ARG MODEL_NAME
ARG DEVICE_TYPE
WORKDIR /workspace

ARG FOLDER_NAME=transformers
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout cc5c061e346365252458126abb699b87cda5dcc0
 
COPY ./${MODEL_NAME}/requirements .

RUN if [ $DEVICE_TYPE == '_mlu' ]; then \
    source /torch/venv3/pytorch/bin/activate && python /torch/src/catch/tools/torch_gpu2mlu.py -i /workspace/${FOLDER_NAME}/; \
    fi

RUN if [ "$DEVICE_TYPE" == '_mlu' ]; then \
        source /torch/venv3/pytorch/bin/activate && pip install --upgrade pip && pip install -r requirements; \
    else \
        pip install --upgrade pip && pip install -r requirements; \
    fi

WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}

COPY ./${MODEL_NAME}/bert_diff_mlu.patch .
COPY ./${MODEL_NAME}/bert_diff_gpu.patch . 

RUN if [ "$DEVICE_TYPE" == '_mlu' ]; then \
        patch -p1 < bert_diff_mlu.patch; \
    else \
        patch -p1 < bert_diff_gpu.patch; \
    fi

WORKDIR /workspace/${FOLDER_NAME}${DEVICE_TYPE}/examples/legacy/question-answering