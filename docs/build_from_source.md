This BKC is for the full building process for Triton. It offers a guide to setup and run Triton models on XPU platform from the source.

- [1. Overview:](#1-overview)
- [2. Env Setting](#2-env-setting)
  - [2.1. XPU Specific env setting](#21-xpu-specific-env-setting)
- [3. Install Dependencies](#3-install-dependencies)
  - [3.1. Install using pip](#31-install-using-pip)
  - [3.2. Build Triton](#32-build-triton)
- [4. Troubleshooting](#4-troubleshooting)
- [5. Triton Test Process](#5-triton-test-process)
- [6. Appendix](#6-appendix)
  - [6.1. Build from the Source](#61-build-from-the-source)
    - [6.1.1. Custom LLVM](#611-custom-llvm)
      - [6.1.1.1. Other optional steps](#6111-other-optional-steps)
    - [6.1.2. PyTorch](#612-pytorch)
    - [6.1.3. Intel® Extension for PyTorch\*](#613-intel-extension-for-pytorch)
  - [6.2. ToolChain](#62-toolchain)
  - [6.3. Driver \& System Install](#63-driver--system-install)



# 1. Overview:

For CUDA Platform:
- LLVM
- PyTorch
- Triton

For XPU Platform:
In addition to CUDA's requirements, you also need
- Intel® oneAPI Base Toolkit/driver (See Appendix for more detail)
- Intel® Extension for PyTorch\*


For all the cases, CUDA/XPU Shares the same building repo as well as the building steps. Thus, unless explicitly pointed out, the build process should fit both CUDA/XPU.

This BKC will be arranged into `Env Setting`, `Build Process`, `TroubleShooting`, `Test Process`, and `Appendix`. If you encounter any build errors, please refer to the troubleshooting section first.

# 2. Env Setting
We recommend using a conda environment for setup. You could skip this part if it is familiar to you. The Python version supports 3.x, you could use any version you like.

```Bash
$conda create -n triton-env python=3.10
$conda activate triton-env
```
For proxy settings, please refer to the appendix.

It is recommended that you set the proxy if needed.
```
export https_proxy=your_proxy
export http_proxy=your_proxy
```

## 2.1. XPU Specific env setting
You need to specify Intel's GPU toolchain. Normally they are in the Intel® oneAPI Base Toolkit. You could refer to the [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide) for the latest updated version.

A sample env setting for oneAPI would be like this:

```Bash
# Sample env setting
source ~/intel/oneapi/compiler/latest/env/vars.sh
source ~/intel/oneapi/mkl/latest/env/vars.sh
source ~/intel/oneapi/dnnl/latest/env/vars.sh

export MKL_DPCPP_ROOT=${HOME}/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${MKL_DPCPP_ROOT}/lib:${MKL_DPCPP_ROOT}/lib64:${MKL_DPCPP_ROOT}/lib/intel64:${LIBRARY_PATH}

# The AOT_DEVLIST should be set according to your device.
# export USE_AOT_DEVLIST='ats-m150'
# export USE_AOT_DEVLIST='pvc'
source ~/intel/oneapi/tbb/latest/env/vars.sh

# Helps to build quicker
export BUILD_SEPARATE_OPS=ON
```

You could store it in a file like `env.sh`, then call the `source env.sh` to activate it next time. Note that please source oneAPI after PyTorch is built, there are known bugs when using oneAPI's compiler to build PyTorch.


# 3. Install Dependencies

It is recommended to install PyTorch and Intel® Extension for PyTorch\* using `pip`. Or you could build them from the source.

**Note**: As we are in an early stage and are changing rapidly. It is highly recommended to build everything from the source following Section [6.1. Build from the Source](#61-build-from-the-source).

## 3.1. Install using pip
Please follow the [Installation Guide](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installations/linux.html) to install the necessary packages. Please be sure that the PyTorch version is from [Intel](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installations/linux.html#pytorch-intel-extension-for-pytorch-version-mapping), rather than from the [stock PyTorch](https://pytorch.org/get-started/locally/)

The difference between these two PyTorch is that the Intel version has some private [torch patches](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master/torch_patches) applied. Those patches are upstreamed to stock PyTorch's master branch, but may not in the current stock PyTorch Release.



## 3.2. Build Triton

```
git clone https://github.com/openai/triton triton
```


Install pybind11 if it is missing:

```Bash
pip install pybind11
```

Build Triton

```Bash
cd triton
git submodule sync
git submodule update --init --recursive --jobs 0
# check to latest third_party backend if needed
cd third_party/intel_xpu_backend
git checkout main && git pull
# check triton commit to verified working commit
cd ../..
git checkout `cat third_party/intel_xpu_backend/triton_hash.txt`
# cd to python folder
cd python
TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py develop
```


# 4. Troubleshooting
See page on

https://github.com/intel/intel-xpu-backend-for-triton/wiki/Possible-Build-Bugs

# 5. Triton Test Process

For the XPU test, Please refer to the doc in the `docs` folder.

# 6. Appendix


## 6.1. Build from the Source
If you wish to build everything from the source, you could follow this recipe.


### 6.1.1. Custom LLVM

Note that there is no need to build LLVM by yourself. If you have something specific purpose like use Debug version of LLVM, you could use refer to the following process, or you could simply use the script [build_llvm.sh](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/.github/scripts/build_llvm.sh).

```Bash
# Use the custom llvm repo for now
git clone -b xpu_llvm_rebase https://github.com/leizhenyuan/llvm-project.git
```

Before build, make sure the related dependencies are installed in your system. See [LLVM-requirements](https://llvm.org/docs/GettingStarted.html#requirements) for detail. One thing worth noticing is that zlib is required for Triton, thus you should install it first.

```Bash
sudo apt-get install zlib1g-dev
```

```Bash
# Note: Choose the branch you need!
cd llvm
mkdir build && cd build
# Please Change DCMAKE_BUILD_TYPE according to your need
cmake ../llvm -G Ninja  -DLLVM_ENABLE_PROJECTS="mlir" -DCMAKE_BUILD_TYPE=Release -DLLVM_USE_LINKER=gold  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU"
ninja all
```

This should build LLVM and all its related targets. You could check those files under the `build` folder.

In case there are build errors, please refer to the [troubleshooting](#4-troubleshooting) page.

Triton build will use the `TRITON_INTEL_LLVM_DIR` for specific LLVM dir. If you choose your own LLVM, you should do the following:

```Bash
export TRITON_INTEL_LLVM_DIR={abs_path_to}/llvm/build/
```


#### 6.1.1.1. Other optional steps


In a later process, if you need something like `LLVM_LIBRARY_DIR`, you could export it using something like:

```Bash
# Not required for triton, example only
export LLVM_LIBRARY_DIR=.../llvm/build/lib
```

For some reason, if you wish to use `mlir-opt` or other binaries, it is recommended to install the LLVM to another directory, you could take the following steps:

```Bash
# Not required for triton, example only
# Your local install folder, could be anything you like
mkdir /home/user/local/llvm-install-folder
# In llvm/build
cmake -DCMAKE_INSTALL_PREFIX=/home/user/local/llvm-install-folder -P cmake_install.cmake
```

Export the path:

```Bash
# Not required for Triton, example only
export PATH=/home/gta/tongsu/llvm/build/bin:${PATH}
```


### 6.1.2. PyTorch

Note that if you build PyTorch from the source, all [torch patches](https://github.com/intel/intel-extension-for-pytorch/tree/xpu-master/torch_patches) must applied and built with `_GLIBCXX_USE_CXX11_ABI=1` enabled.


First clone PyTorch with `v2.0.1` tag and intel-extension-for-pytorch repo with `xpu-master` branch:

```Bash
git clone -b v2.0.1 https://github.com/pytorch/pytorch/
git clone -b xpu-master https://github.com/intel/intel-extension-for-pytorch
```
**Important** : Apply the torch patches:

```Bash
cd pytorch
git apply ../intel-extension-for-pytorch/torch_patches/*.patch
```
**Important** Be sure to build PyTorch with  `_GLIBCXX_USE_CXX11_ABI=1` enabled:

```Bash
export _GLIBCXX_USE_CXX11_ABI=1
```

```Bash
git submodule sync
git submodule update --init --recursive --jobs 0

conda install cmake ninja mkl mkl-include
pip install -r requirements.txt
```

Then build PyTorch. Note that please make sure the oneAPI is not sourced at this step. Just use GCC to build PyTorch instead of icpx (The compiler in oneAPI).

```Bash
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
```

Test if PyTorch is installed without error:

```Bash
cd ..
python -c "import torch;print(torch.__version__)"
```

### 6.1.3. Intel® Extension for PyTorch\*

```Bash
cd intel-extension-for-pytorch
git submodule sync
git submodule update --init --recursive --jobs 0
pip install -r requirements.txt
```
First source [env.sh](#21-xpu-specific-env-setting) mentioned earlier:

```Bash
source env.sh
python setup.py bdist_wheel
pip install dist/*.whl
```

Test if intel-extension-for-pytorch is installed without error:

```Bash
cd ..
python -c "import torch;import intel_extension_for_pytorch as ipex;print(ipex.__version__)"
```

## 6.2. ToolChain
For driver and Toolchains, you could go to this page for the latest alignment.

https://wiki.ith.intel.com/display/PyTorchdGPU/Tool+Chain+and+BKC+alignment

## 6.3. Driver & System Install

For the Installation guide about the system, you could refer to:

https://dgpu-docs.intel.com/driver/installation.html
