ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.2.1-torch1.6-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

COPY ./scratch_DLRM_2mlu.sh /torch/src/pytorch_models/Training/Recommendation/DLRM/cambricon/
COPY ./scratch_DLRM_2mlu_test.sh /torch/src/pytorch_models/Training/Recommendation/DLRM/cambricon/

WORKDIR /torch/src/pytorch_models/Training/Recommendation