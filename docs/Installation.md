This Doc contains information for building `intel-xpu-backend-for-triton`.


- [Overview:](#overview)
- [Env Setting](#env-setting)
  - [XPU Specific env Preparation](#xpu-specific-env-preparation)
- [Build Process](#build-process)
  - [llvm](#llvm)
  - [PyTorch](#pytorch)
  - [IPEX](#ipex)
  - [Triton](#triton)
- [Troubleshooting](#troubleshooting)
- [Triton Test Process](#triton-test-process)

# Overview:

For XPU Platform, it needs the following components for build:
- llvm
- pytorch
- triton
- oneAPI Basekit / driver (See Appendix for more detail)
- Intel-Extension-For-Pytorch

intel-xpu-backend-for-triton serves as a third-party package for [triton](https://github.com/openai/triton). It couldn't build without triton. One should build from triton repo, instead of building from intel-xpu-backend-for-triton.

This BKC will be arranged into `Env Setting`, `Build Process`, `TroubleShooting` , `Test Process` and `Appendix`. If you encountered any build errors, please refer to the troubleshooting section first.

# Env Setting
We recommend using a conda environment for setup. You could skip this part if it is familiar to you. The Python version supports 3.x, you could use any version you like.

```Bash
$conda create -n triton-env python=3.10
$conda activate triton-env
```

## XPU Specific env Preparation
You need to specify the Intel's GPU toolchain. One could refer to the [Installation Guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#preparations) for the most updated version, which specifies the driver and toolkit needed. Basically, one needs to install necessary driver and oneAPI toolkit.


# Build Process

## llvm

If you wish to use custom LLVM, it is recommended to use llvm commit newer than (TODO).

## PyTorch

Install PyTorch following documents at [PyTorch](https://pytorch.org/get-started/locally/). The PyTorch version should newer than 2.0.1.

## IPEX

Follow instructions at [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch/tree/master). The version should at least 2.0.1, and match PyTorch version.

```Bash
$git clone -b xpu-master https://github.com/intel/intel-extension-for-pytorch/tree/master
$cd intel-extension-for-pytorch
$git submodule sync
$git submodule update --init --recursive --jobs 0
$pip install -r requirements.txt
$python setup.py develop
```

## Triton
intel-xpu-backend-for-triton serves as a third-party package for [triton](https://github.com/openai/triton). One should build from the triton repo, instead of building from intel-xpu-backend-for-triton.

```Bash
# Clone openai/triton
$git clone https://github.com/openai/triton.git
# Clone submodules
$git submodule sync
$git submodule update --init --recursive --jobs 0
$cd triton
```

[Optional] set the env flags so that triton build uses `pybind11` and `llvm` in the environment instead of re-download it.
Normally, this is not need.

```Bash
# Optional Steps
# Install pybind11 if not exist
$conda install -c conda-forge pybind11
# Get from conda envs. You could check with `which python`
$export PYBIND11_SYSPATH=../envs/../lib/python3.10/site-packages/pybind11/
# Formerly build llvm path
$export LLVM_SYSPATH=../llvm/build/
```

Now Build triton:

```Bash
$cd python
$TRITON_BUILD_WITH_XPU_SUPPORT=1 python setup.py develop
```

# Troubleshooting
TODO: Add common troubleshooting page

# Triton Test Process
For XPU test, Please refer to page on wiki:
TODO: Add Test instructions



