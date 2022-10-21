ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.5.0-torch1.9-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

WORKDIR /

RUN source /torch/venv3/pytorch/bin/activate && pip install -U openmim && pip install mmcv-full==1.6.0
RUN git clone https://github.com/Qiza-lyhm/mmcls_mlu.git && cd mmcls_mlu && pip install -v -e .

WORKDIR /mmcls_mlu

COPY resnet50_cifar100_16*b496.py configs
