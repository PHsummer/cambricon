FROM nvcr.io/nvidia/pytorch:21.07-py3

WORKDIR /workspace
RUN git clone https://github.com/open-mmlab/OpenPCDet.git && cd OpenPCDet && git checkout 274c90c55364ab87e15cd27b3e19005b795ae503 && pip install spconv-cu114 

WORKDIR /workspace/OpenPCDet
RUN python setup.py develop && pip install nuscenes-devkit==1.0.5
RUN apt update && apt install libgl1-mesa-glx -y

COPY ./cbgs_pp_multihead.yaml /workspace
COPY ./cbgs_pp_multihead_nds5817.yaml /workspace
