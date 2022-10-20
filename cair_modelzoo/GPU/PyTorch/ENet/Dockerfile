ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/davidtvs/PyTorch-ENet.git && cd PyTorch-ENet && git checkout e17d404e2f649a3476eabe39f8a05e5eb77c55fd

WORKDIR /workspace/PyTorch-ENet
RUN pip install --no-cache-dir -r requirements.txt

COPY ./PyTorch/ENet/modify.patch .