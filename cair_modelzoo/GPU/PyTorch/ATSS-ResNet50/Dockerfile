ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/cair_modelzoo/gpu_golden_bench:atss-r50-21.06-pytorch-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN rm -r /workspace/mmdetection
RUN git clone https://github.com/open-mmlab/mmdetection.git && cd mmdetection && git checkout ca11860f4f3c3ca2ce8340e2686eeaec05b29111

WORKDIR /workspace/mmdetection
# RUN pip install terminaltables==3.1.10 -i https://pypi.tuna.tsinghua.edu.cn/simple/
# RUN pip install mmcv-full==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple/

COPY ./PyTorch/ATSS-ResNet50/modify.patch .