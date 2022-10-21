ARG FROM_IMAGE_NAME=nvcr.io/nvidia/tensorflow:22.01-tf1-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/ChenglongChen/tensorflow-DeepFM.git && cd tensorflow-DeepFM && git checkout a43dd5ff1f61a275c8a6e0fd659e200af64093cc

WORKDIR /workspace/tensorflow-DeepFM//example
RUN git clone https://github.com/JianGoForIt/YellowFin &&  cd YellowFin && git checkout 847d9948a3707dde501318ba3d31db2c4c7e3e61
WORKDIR /workspace/tensorflow-DeepFM//example/YellowFin
RUN cp tuner_utils/yellowfin.py tuner_utils/yellowfin_test.py tuner_utils/debug_plot.py ../

WORKDIR /workspace/tensorflow-DeepFM/example
COPY ./TensorFlow/DeepFM/data /workspace/tensorflow-DeepFM/example/data/
RUN pip uninstall numpy && pip install -U numpy==1.18.5
