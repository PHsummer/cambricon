ARG BASE_IMAGE=yellow.hub.cambricon.com/pytorch/pytorch:v1.3.0-torch1.6-ubuntu18.04
FROM ${BASE_IMAGE}

RUN mkdir -p /root/.pip && \
    # echo "[global]" > /root/.pip/pip.conf && \
    # echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf && \
    # echo "trusted-host = mirrors.aliyun.com" >> /root/.pip/pip.conf
    echo "[global]" > /root/.pip/pip.conf && \
    echo "index-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> /root/.pip/pip.conf && \
    echo "trusted-host = mirrors.tuna.tsinghua.edu.cn" >> /root/.pip/pip.conf

# Python 3.6, 3.7 is supported by Ubuntu Bionic out of the box
ARG PY_VERSION=3.6
ENV PYTHON_VERSION_NO=${PY_VERSION}

ENV LANG C.UTF-8

# # Install the time package
# ARG DEBIAN_FRONTEND=noninteractive
# ARG TZ=Asia/Shanghai
# RUN apt-get -y update && \
#     apt-get install -y tzdata && \
#     ln -snf /usr/share/zoneinfo/${TZ} /etc/localtime && echo $TZ > /etc/timezone && \
#     dpkg-reconfigure --frontend ${DEBIAN_FRONTEND} tzdata && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# Install common software
RUN apt-get -y update && \
    apt-get -y install \
        apt-utils \
        build-essential \
        software-properties-common \
        wget \
        curl \
        gcc \
        ssh \
        git \
        vim \
        swig \
        zip unzip \
        gzip \
        rsync \
        virtualenv \
        zlib1g-dev \
        libffi-dev \
        libssl-dev \
        libopenblas-dev \
        libibverbs-dev \
        libgtest-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libboost-regex-dev \
        tcl-dev \
        tk-dev \
        python${PYTHON_VERSION_NO} \
        python${PYTHON_VERSION_NO}-dev \
        python${PYTHON_VERSION_NO}-distutils \
        python3-pip \
        *-mpi-*- \
        *mpich*- \
        *openmpi*-  \
        pdsh \
        python3-pip \
        tcsh ca-certificates \
        iputils-ping net-tools ethtool \
        perl lsb-release python-libxml2 \
        iproute2 pciutils libnl-route-3-dev \
        kmod libnuma1 lsof \
        libelf1 automake libglib2.0-0 \
        autoconf graphviz chrpath flex libnl-3-dev m4 \
        debhelper autotools-dev gfortran libltdl-dev \
        libusb-1.0-0 bison udev libfuse2 dpatch && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.19.5/cmake-3.19.5-Linux-x86_64.tar.gz && \
    tar xf cmake-3.19.5-Linux-x86_64.tar.gz && \
    mv cmake-3.19.5-Linux-x86_64 /usr/local/cmake && \
    rm /tmp/cmake*

ENV PATH=/usr/local/cmake/bin:$PATH

RUN cd /usr/src/gtest && \
    cmake CMakeLists.txt && \
    make -j && \
    cp libgtest* /usr/lib && \
    make clean

# # Install Open MPI
# ARG OPENMPI_VERSION
#
# ENV MPI_HOME=/usr/local/openmpi
# ENV PATH=${MPI_HOME}/bin:$PATH
# ENV LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
#
# RUN mkdir -p $MPI_HOME
# RUN cd /tmp && \
#     apt-get -y install bzip2 && \
#     wget http://gitlab.software.cambricon.com/neuware/platform/extra-dependencies/-/raw/master/mpi/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
#     tar -jxvf openmpi-${OPENMPI_VERSION}.tar.bz2 && \
#     cd openmpi-${OPENMPI_VERSION} && \
#     ./configure --prefix=$MPI_HOME --enable-orterun-prefix-by-default && \
#     make -j && \
#     make install && \
#     rm /tmp/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
#     rm -rf /tmp/openmpi-${OPENMPI_VERSION}

# # Install OpenSSH for MPI to communicate between containers
# RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
#     mkdir -p /var/run/sshd
#
# # Allow OpenSSH to talk to containers without asking for confirmation
# RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
#     echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
#     mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config


# Install InfiniBand driver
ARG MOFED_VERSION=5.3-1.0.0.1
ARG OS_NAME=ubuntu
ARG OS_VERSION=18.04

# # Download and install Mellanox OFED 5.3-1.0.0.1 for Ubuntu 18.04
# RUN cd /tmp && \
#     wget http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VERSION}/MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_NAME}${OS_VERSION}-x86_64.tgz && \
#     tar -xvf MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_NAME}${OS_VERSION}-x86_64.tgz && \
#     MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_NAME}${OS_VERSION}-x86_64/mlnxofedinstall --user-space-only --without-fw-update --skip-unsupported-devices-check -q --force && \
#     rm MLNX_OFED_LINUX-${MOFED_VERSION}-${OS_NAME}${OS_VERSION}-x86_64.tgz

#install virtual env and libraries needed by GLM
ARG PY_VENV=/torch/venv3/pytorch


# Install cndsp
ARG CNDSP_BRANCH=master
RUN cd /tmp && \
    git clone http://gitlab.software.cambricon.com/neuware/platform/cndsp.git -b ${CNDSP_BRANCH} --depth 1 && \
    cd /tmp/cndsp && \
    git submodule update --init --recursive && \
    ls /tmp/cndsp/deps/deepspeed/deepspeed && \
    . ${PY_VENV}/bin/activate && \
    bash build.sh -t install

WORKDIR /
COPY ./GLM /

# Install required modules by GLM model
WORKDIR /
RUN bash -c "source ${PY_VENV}/bin/activate && pip install -r requirements.txt"

RUN echo "source ${PY_VENV}/bin/activate" >> ~/.bashrc
