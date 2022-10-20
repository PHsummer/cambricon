ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.01-tf1-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/kyzhouhzau/BERT-NER.git && cd BERT-NER && git checkout 0f77e478872453df51cd3c65d1a39b12d9617f9d

RUN cd /workspace/BERT-NER && git clone https://github.com/google-research/bert.git 
RUN cd /workspace/BERT-NER/bert && git checkout eedf5716ce1268e56f0a50264a88cafad334ac61
WORKDIR /workspace/BERT-NER
COPY ./TensorFlow/BERT-CRF/cased_L-12_H-768_A-12 /workspace/BERT-NER/cased_L-12_H-768_A-12/
