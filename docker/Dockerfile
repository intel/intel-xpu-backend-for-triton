# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE} AS dev-base

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    apt-utils \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gnupg2 \    
    gpg-agent \
    libsm6 \
    libxext6 \
    pybind11-dev \
    wget \
    vim \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache
ENV PATH /opt/conda/bin:$PATH

FROM dev-base AS conda
ARG PYTHON_VERSION=3.10
RUN curl -fsSL -v -k -o ~/miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-py39_23.9.0-0-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    # Fix conda install issue
    sed -i "s;#!/bin/sh;#!/bin/bash;" ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda config --set channel_priority strict && \
    /opt/conda/bin/conda config --append channels conda-forge && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} \
    astunparse \
    cffi \
    cmake>=3.13.0 \
    dataclasses \
    future \
    libstdcxx-ng \
    mkl-include \
    mkl==2021.4.0 \
    ninja \
    numpy \
    requests \
    typing_extensions \
    pyyaml \
    setuptools && \
    /opt/conda/bin/conda clean -ya && \
    # Fix GLIBCXX version issue
    rm -f /opt/conda/lib/libstdc++.so.6

FROM dev-base AS build
ARG PT_REPO=https://github.com/pytorch/pytorch
ARG PT_BRANCH=release/2.1
ARG PT_COMMIT=209f2fa8ff86652f67d75c2f19bf9cb9942fd018
ARG IPEX_REPO=https://github.com/intel/intel-extension-for-pytorch
ARG IPEX_BRANCH=xpu-main-pre
ARG IPEX_COMMIT=7980a37028023037b4f0b47617c5fc3343a6d09b
COPY --from=conda /opt/conda /opt/conda
WORKDIR /opt
RUN git clone https://github.com/openai/triton triton && \
    cd triton && git submodule sync && git submodule update --init --recursive && \
    cd third_party/intel_xpu_backend && git checkout main && git pull && cd ../.. && \
    git checkout `cat third_party/intel_xpu_backend/triton_hash.txt` && cd .. && \
    git clone -b ${PT_BRANCH} ${PT_REPO} && \
    git clone -b ${IPEX_BRANCH} ${IPEX_REPO} && \
    cd pytorch && git checkout ${PT_COMMIT} && git submodule sync && git submodule update --init --recursive && \
    git apply ../intel-extension-for-pytorch/torch_patches/*.patch && \
    rm -rf ../intel-extension-for-pytorch && \
    python setup.py bdist_wheel && pip install dist/*.whl && cd .. && rm -rf pytorch
ARG BASEKIT_URL=https://registrationcenter-download.intel.com/akdlm/IRC_NAS/20f4e6a1-6b0b-4752-b8c1-e5eacba10e01/l_BaseKit_p_2024.0.0.49564_offline.sh
RUN export TERM=dumb && \
    wget ${BASEKIT_URL} -P /opt && \
    chmod +x /opt/l_BaseKit_*.sh && \
    cd /opt && mkdir -p /opt/intel/oneapi && \
    sh ./l_BaseKit_*.sh -a --cli --silent --eula accept --install-dir /opt/intel/oneapi && \
    rm -rf /opt/l_BaseKit_*.sh

# /usr/bin/ld: cannot find -lze_loader: No such file or directory
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
    tee /etc/apt/sources.list.d/intel-gpu-jammy.list
RUN apt-get update && apt-get install --no-install-recommends -y \
    intel-opencl-icd \
    intel-level-zero-gpu \
    level-zero \
    level-zero-dev \
    && rm -rf /var/lib/apt/lists/*
ENV PATH /opt/conda/bin:$PATH  
    
RUN chmod +r -R /opt/intel/oneapi && \
    . /opt/intel/oneapi/setvars.sh && \
    export MKL_DPCPP_ROOT=${MKLROOT} && \
    export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH} && \
    export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:$LIBRARY_PATH && \
    export USE_AOT_DEVLIST='12.55.8' && \
    export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 && \
    export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=0 && \
    export ExperimentalCopyThroughLock=1 && \
    # /usr/bin/ld: cannot find -lze_loader: No such file or directory
    export LIBZE_PATH=/usr/lib/x86_64-linux-gnu/libze_loader.so && \
    git clone -b ${IPEX_BRANCH} ${IPEX_REPO} && \
    cd intel-extension-for-pytorch && git checkout ${IPEX_COMMIT} && \
    git submodule sync && git submodule update --init --recursive && \
    pip install -r requirements.txt && \
    python setup.py bdist_wheel && pip install dist/*.whl && cd .. && \
    rm -rf intel-extension-for-pytorch && \
    cd triton/python && python setup.py clean && \
    TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py bdist_wheel && pip install dist/*.whl

FROM dev-base AS image
COPY --from=build /opt/conda /opt/conda
COPY --from=build /opt/intel /opt/intel
COPY --from=build /opt/triton /opt/triton
COPY --from=build /usr/lib /usr/lib
RUN . /opt/intel/oneapi/setvars.sh
WORKDIR /opt/triton