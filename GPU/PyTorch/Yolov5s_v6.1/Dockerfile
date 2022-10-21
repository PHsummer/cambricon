FROM nvcr.io/nvidia/pytorch:21.06-py3

WORKDIR /workspace
RUN git clone https://github.com/ultralytics/yolov5.git  && cd yolov5 && git checkout 3752807c0b8af03d42de478fbcbf338ec4546a6c

WORKDIR /workspace/yolov5
RUN apt-get update && apt-get -y install libgl1-mesa-glx
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && pip install opencv-python==4.5.5.64 && pip install seaborn
COPY ./PyTorch/Yolov5s_v6.1/yolov5_6.1.patch .
RUN git apply --ignore-space-change --ignore-whitespace  --whitespace=fix yolov5_6.1.patch
