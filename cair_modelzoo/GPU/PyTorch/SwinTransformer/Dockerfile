ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3 
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/microsoft/Swin-Transformer.git && cd Swin-Transformer && git checkout 78cec9ac5c746c7e72305a9a24716ddb3fcc043
RUN pip install timm==0.4.12 yacs termcolor

WORKDIR /workspace/Swin-Transformer
COPY ./PyTorch/SwinTransformer/modify.patch .
RUN git apply --ignore-space-change --ignore-whitespace --whitespace=nowarn modify.patch
