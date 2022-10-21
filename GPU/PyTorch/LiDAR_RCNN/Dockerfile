ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:21.06-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace
RUN git clone https://github.com/TuSimple/LiDAR_RCNN.git && cd LiDAR_RCNN && git checkout 89c5fefd46563c15f31b8cda76e7ab09e2ddca0d

WORKDIR /workspace/LiDAR_RCNN
RUN git clone https://github.com/pybind/pybind11.git && cd pybind11 && mkdir build && cd build && cmake .. && make -j && make install
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install ninja-build libeigen3-dev
RUN python setup.py develop --user
RUN pip3 install --upgrade pip && pip3 install waymo-open-dataset-tf-2-4-0 --user -i https://pypi.tuna.tsinghua.edu.cn/simple && pip3 install --upgrade numpy

COPY ./PyTorch/LiDAR_RCNN/modify.patch .
RUN git apply --ignore-space-change --ignore-whitespace --reject modify.patch