ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone  https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git && cd pytorch-CycleGAN-and-pix2pix && git checkout 003efc4c8819de47ff11b5a0af7ba09aee7f5fc1

WORKDIR /workspace/pytorch-CycleGAN-and-pix2pix
RUN pip install --no-cache-dir -r requirements.txt
RUN bash ./datasets/download_pix2pix_dataset.sh facades
