ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}


WORKDIR /workspace
RUN git clone https://github.com/speechbrain/speechbrain.git && cd speechbrain && git checkout 98f90f82acc327a8180f3591135a18e278d3e0e2 && sed -i 's|torch>=1.8.0,<=1.10.1|torch==1.9.0|g' ./requirements.txt && sed -i 's|torchaudio>=0.8.0,<=0.10.1|torchaudio==0.9.0|g' ./requirements.txt

WORKDIR /workspace/speechbrain
RUN pip install -r requirements.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -e .


WORKDIR /workspace/speechbrain/recipes/VoxCeleb/SpeakerRec
COPY ./PyTorch/ECAPA_TDNN/modify.patch .
COPY ./PyTorch/ECAPA_TDNN/train_ecapa_tdnn.yaml .
