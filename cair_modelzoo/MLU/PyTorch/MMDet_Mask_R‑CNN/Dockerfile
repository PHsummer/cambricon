ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.2.1-torch1.6-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

COPY ./scratch_mask_rcnn_2mlu.sh /torch/src/mmdetection/scripts/
COPY ./scratch_mask_rcnn_2mlu_test.sh /torch/src/mmdetection/scripts/

WORKDIR /torch/src/mmdetection/scripts