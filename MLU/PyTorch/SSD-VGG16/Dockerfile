ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.2.1-torch1.6-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

COPY ./scratch_2mlu_amp.sh /torch/src/pytorch_models/Training/Detection/ssd_vgg16/cambricon/

WORKDIR /torch/src/pytorch_models/Training/Detection
