ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/cair_modelzoo/mlu-benchmark:bert-large-catch1.1.2-1-x86_64-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

WORKDIR /
COPY ./mlperf_bert /

WORKDIR /mlperf_bert
RUN /bin/bash -c "source /torch/venv3/pytorch/bin/activate && cd mhalib && python setup.py install"
RUN /bin/bash -c "source /torch/venv3/pytorch/bin/activate && cd mlperf-logging && python setup.py install"
