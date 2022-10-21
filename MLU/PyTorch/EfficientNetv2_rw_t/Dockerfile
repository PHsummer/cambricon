ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.5.0-torch1.9-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
COPY ./pytorch-image-models_mlu ./pytorch-image-models_mlu

WORKDIR /workspace/pytorch-image-models_mlu
COPY ./run_scripts/*.sh .
