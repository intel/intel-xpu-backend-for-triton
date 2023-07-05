- [Overview:](#overview)
- [Env Setting](#env-setting)
  - [XPU Specific env Preparation](#xpu-specific-env-preparation)
- [Build Process](#build-process)
  - [LLVM](#llvm)
  - [PyTorch](#pytorch)
  - [intel-extension-for-pytorch](#intel-extension-for-pytorch)
  - [Triton](#triton)
- [Appendix](#appendix)
  - [Common Bugs](#common-bugs)
  - [Custom LLVM](#custom-llvm)


# Overview:

intel-xpu-backend-for-triton serves as a third-party package for [triton](https://github.com/openai/triton). It couldn't build without triton. One should build from triton repo, instead of building from intel-xpu-backend-for-triton.

This BKC will be arranged into `Env Setting`, `Build Process`, `TroubleShooting` , `Test Process` and `Appendix`. If you encountered any build errors, please refer to the [Common Bugs](#common-bugs) section first.

# Env Setting
We recommend using a conda environment for setup. You could skip this part if it is familiar to you. The Python version supports 3.x, you could use any version you like.

```Bash
$conda create -n triton-env python=3.10
$conda activate triton-env
```

## XPU Specific env Preparation
You need to specify the Intel's GPU toolchain. One could refer to the [Installation Guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#preparations) for the most updated version, which specifies the driver and toolkit needed. Basically, one needs to install necessary driver and oneAPI toolkit.

It is recommended to set an `env_triton.sh` for later use.

```Bash
# Sample env setting
source ~/intel/oneapi/compiler/latest/env/vars.sh
source ~/intel/oneapi/mkl/latest/env/vars.sh
source ~/intel/oneapi/dnnl/latest/env/vars.sh

export MKL_DPCPP_ROOT=${HOME}/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LIBRARY_PATH}

# The AOT_DEVLIST should be set according to your device. It helps for running quicker.
# Please refer to
# https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/AOT.html
export USE_AOT_DEVLIST='pvc'
source ~/intel/oneapi/tbb/latest/env/vars.sh

# Helps to build quicker
export BUILD_SEPARATE_OPS=ON
```
You could store it into a file like `env_triton.sh`, then call the `source env_triton.sh` to activate it next time.

# Build Process

For XPU Platform, the build process is as follows, note that we are in an early dev stage, thus PyTorch and
Intel-Extension-For-Pytorch all need to be built from the source.

- LLVM
- PyTorch
- Intel-Extension-For-Pytorch
- triton


## LLVM

** Important Note**:

We are at an active development stage, thus there are many local changes waiting to upstream to stock llvm. Currently, one need to use the custom llvm. Please follow the  detail in [Custom LLVM](#custom-llvm) section first. In later version, the default triton-specified version of llvm would work.

If one wishes to use custom LLVM, one needs to specify a recent version to make functionality work. It is highly recommended to use the default triton's version later.


## PyTorch

Install PyTorch following documents at [PyTorch](https://pytorch.org/get-started/locally/). The PyTorch version should newer than 2.0.1. Because there are patches need to be applied for building IPEX, the PyTorch needs to be built from the source:

```Bash
conda install cmake ninja mkl mkl-include
conda install -y -c conda-forge libstdcxx-ng

git clone -b v2.0.1 https://github.com/pytorch/pytorch.git
cd pytorch
pip install -r requirements.txt
git submodule sync && git submodule update --init --recursive
git apply ../intel-extension-for-pytorch/torch_patches/*.patch

export _GLIBCXX_USE_CXX11_ABI=1
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

Verify the installation by:

```Bash
python -c "import torch;print(torch.__version__)"
```

## intel-extension-for-pytorch

Follow instructions at [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch/).

Note that you need to first set env as mentioned in [XPU Specific env Preparation](#xpu-specific-env-preparation) first. i.e.,

```Bash
# env_triton.sh is set by former step.
source env_triton.sh
```


```Bash
git clone -b xpu-master https://github.com/intel/intel-extension-for-pytorch/tree/master
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive --jobs 0
pip install -r requirements.txt
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
```

Verify the installation by:

```Bash
python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
```

## Triton

intel-xpu-backend-for-triton serves as a third-party package for [triton](https://github.com/openai/triton). One should build from the triton repo, instead of building from intel-xpu-backend-for-triton.

```Bash
# Clone openai/triton
git clone https://github.com/openai/triton.git
cd triton
# Clone submodules
git submodule sync && git submodule update --init --recursive --jobs 0
```
Since we are at the active development stage, it is recommended to check to latest commit for ntel-xpu-backend-for-triton:

```Bash
cd third_party/intel_xpu_backend
git checkout main && git pull
```


[Optional] set the env flags so that triton build uses `pybind11` and `llvm` in the environment instead of re-download it.
Normally, this is not need.

```Bash
# Optional Steps
# Install pybind11 if wish custom pybind11
conda install -c conda-forge pybind11
# Get from conda envs. You could check with `which python`
export PYBIND11_SYSPATH={your-abs-path-to/site-packages/pybind11}
# Formerly build llvm path
export LLVM_SYSPATH={your-abs-path-to}/llvm/build/
```

Now Build triton:

```Bash
cd {triton-root-dir}
cd python
TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py develop
```
# Appendix
## Common Bugs
If you encountered any problem, please refer to the [Common Bugs](common_bugs.md) first.

## Custom LLVM

**The following step is optional**

Before build, make sure the related dependencies are installed in your system. See [llvm-requirements](https://llvm.org/docs/GettingStarted.html#requirements) for detail. One thing worth noticing is that zlib is required for triton, thus you should install it first.

```Bash
sudo apt-get install zlib1g-dev
```

```Bash
# Build Custom llvm
# Use this private one
git clone -b triton_debug https://github.com/chengjunlu/llvm/

#  Public one [Not for now]
git clone https://github.com/llvm/llvm-project

# Note: Choose the branch you need!
cd llvm
mkdir build && cd build
cmake ../llvm -G Ninja  -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=gold  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"
ninja all
```

For later triton use, one need to specify custom llvm path at:

```Bash
export LLVM_SYSPATH={abs_path_to}/llvm/build/
```
