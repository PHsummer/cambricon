# Docker image
# 

# Set mmdeploy
export MMDEPLOY_VERSION=0.6.0
export TENSORRT_VERSION=8.2.3.0
export PYTHON_VERSION=3.8
export PYTHON_STRING="${PYTHON_VERSION/./""}"

# Set mmdet
cd ./mmdetection
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools openmim onnxruntime-gpu pycuda
mim install mmcv-full==1.6.0
pip install -v -e .
cd ..

# Set mmdeploy
if [ ! -d "./mmdeploy-0.6.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0" ]; then
    wget https://github.com/open-mmlab/mmdeploy/releases/download/v0.6.0/mmdeploy-0.6.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
    tar -zxvf mmdeploy-0.6.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0.tar.gz
fi
cd mmdeploy-0.6.0-linux-x86_64-cuda11.1-tensorrt8.2.3.0
python -m pip install dist/mmdeploy-*-py${PYTHON_STRING}*.whl
python -m pip install sdk/python/mmdeploy_python-*-cp${PYTHON_STRING}*.whl
export LD_LIBRARY_PATH=$(pwd)/sdk/lib:$LD_LIBRARY_PATH
cd ..
apt-get update
apt-get install -y libx11-dev

# wget https://developer.download.nvidia.com/compute/machine-learning/tensorrt/secure/8.2.3.0/tars/TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz?RKV2pb7AFi3JLgvKLQgafP8iC3btWhYryz5wV7LUipkVoFQk6hXmFkiKR3ukr-p-zw3Fm85vFVsmLu6gQiIGVejwReTZbuwmuOm3XuwD7KKG_AKx3SSpdeNXnEgKpIyf-Vq1HjZZUyNxLS0eriX8Gtv6lyX2WBLv5hIQmEm67H9tGoM9l40r1-E0rfQn4WA28XjfhnzVrRWwXZOviIdLfi5UUZqVjhT92kT_PT42sY8&t=eyJscyI6InJlZiIsImxzZCI6IlJFRi13d3cuYmFpZHUuY29tXC9saW5rP3VybD1iOFhfM2ZFSUNSVzZNQ01Pa2VSWGxSR05DQi03anZVYWlrVFZxbzhOeDM4OXJKSU5mb2RMWUFQTXlPMGxoN1BEJndkPSZlcWlkPWE3OTU0OWQyMDAwMmJhNTcwMDAwMDAwNjYyZDdjZGY5In0
# tar -zxvf TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.4.cudnn8.2.tar.gz
# python -m pip install TensorRT-${TENSORRT_VERSION}/python/tensorrt-*-cp${PYTHON_STRING}*.whl
# python -m pip install pycuda
# export TENSORRT_DIR=$(pwd)/TensorRT-${TENSORRT_VERSION}
# export LD_LIBRARY_PATH=${TENSORRT_DIR}/lib:$LD_LIBRARY_PATH

# /opt/conda/lib/python3.8/site-packages/mmdeploy/utils/constants.py
# line14, need to add: 
# "if k == value: return k"
# within "for loop"

cd ./mmdeploy
python tools/check_env.py
cd ..

