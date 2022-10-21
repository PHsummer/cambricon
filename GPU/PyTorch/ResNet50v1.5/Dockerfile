ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/NVIDIA/DeepLearningExamples.git && cd DeepLearningExamples && git checkout 3a8068b651c8ae919281b638166b3ecfa07d22f5

WORKDIR /workspace/DeepLearningExamples/PyTorch/Classification/ConvNets
RUN pip install --no-cache-dir -r requirements.txt