ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/pytorch:v1.5.0-torch1.9-ubuntu18.04
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
COPY ./Yolov5m_v6.0 ./Yolov5m_v6.0

WORKDIR /workspace/Yolov5m_v6.0
RUN source /torch/venv3/pytorch/bin/activate && apt-get update && apt-get -y install libgl1-mesa-glx && pip install -r requirements.txt
