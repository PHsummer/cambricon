ARG FROM_IMAGE_NAME=yellow.hub.cambricon.com/pytorch/daily/x86_64/pytorch:v1.4.0-torch1.9-x86_64-ubuntu18.04-catch_1.9_develop_view_chain_new_2.0-latest
FROM ${FROM_IMAGE_NAME}

WORKDIR /test_espnet
RUN source /torch/venv3/pytorch/bin/activate && pip install typeguard==2.13.3 pyyaml==6.0 humanfriendly==10.0 torch_complex==0.4.3 pytorch_wpe==0.0.1 h5py==3.1.0 kaldiio==2.17.2 bbmix==0.2.1 g2p_en==2.1.0 jamo==0.4.1 sentencepiece==0.1.96 tacotron==0.1.1 espnet-tts-frontend==0.0.3 configargparse tensorboard==2.9.1 numba==0.43.0 llvmlite==0.32.1 editdistance==0.6.0 ci_sdr fast_bss_eval==0.1.4
RUN apt-get install bc sox -y
WORKDIR /test_espnet/espnet_mlu
COPY ./ ./
WORKDIR /test_espnet/espnet_mlu/egs2/aishell/asr1