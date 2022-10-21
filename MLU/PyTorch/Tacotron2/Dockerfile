ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.2.1-torch1.9-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

COPY ./scratch_tacotron2_2mlu_AMP.sh /torch/src/pytorch_models/Training/SpeechSynthesis/Tacotron2andWaveGlow/cambricon/

WORKDIR /torch/src/pytorch_models/Training/SpeechSynthesis
