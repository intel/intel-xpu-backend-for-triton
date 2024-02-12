# Building Triton for Intel GPUs

## Dependencies

### Drivers and OneAPI Toolkit

To build triton from source, we need to start with installing GPU drivers. If you're using Data Center (PVC) GPUs, please follow the instructions from [here](https://dgpu-docs.intel.com/driver/installation.html#install-steps) of your Linux distribution. For Intel Arc (client) GPUs, here are the [instructions](https://dgpu-docs.intel.com/driver/client/overview.html).

We also need to install OneAPI Basekit with version >= 2024.0.1. Please follow the instructions from [here](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) to install it.


### Python and Virtual Environment

Next, we install python3 and venv in our system. For Ubuntu 22.04, follow these instructions

```
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
```

In other Linux distributions, you might have to follow different instructions. Note that python needs to 3.9 or 3.10 for other distrubutions.


## Build from Source

First clone this repository:

```
git clone https://github.com/intel/intel-xpu-backend-for-triton.git -b llvm-target
cd intel-xpu-backend-for-triton
```

Set up dpc++ environment variables. This works on Ubuntu, you might have to change the directories based on installation.

```
export DPCPPROOT=/opt/intel/oneapi/compiler/latest
export MKLROOT=/opt/intel/oneapi/mkl/latest/
export ONEAPIROOT=/opt/intel/oneapi/
source ${DPCPPROOT}/env/vars.sh
source ${MKLROOT}/env/vars.sh
source ${ONEAPIROOT}/setvars.sh
```

Now let's create a base dir where build will happen and start the build there

```
export BASE=$HOME/triton-build
mkdir $BASE
scripts/compile-triton.sh --venv
```

This should build triton in the `$BASE` dir above. It will also pull a fork of LLVM and build it - so get a coffee while it builds. Once build is complete you can test if import works:

```
$ source $BASE/intel-xpu-backend-for-triton/.venv/bin/activate
$ python
Python 3.10.12 (main, Nov 20 2023, 15:14:05) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import intel_extension_for_pytorch
>>> import triton
```

This should not throw any error!
