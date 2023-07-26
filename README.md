# Intel® XPU Backend for Triton\*

This is the development repository of Intel® XPU Backend for Triton\*, a new [OpenAI Triton](https://github.com/openai/triton) backend for Intel® GPUs. Triton is a language and compiler for writing highly efficient custom Deep-Learning primitives. The aim of Triton\* is to provide an open-source environment to write fast code at higher productivity than CUDA, but also with higher flexibility than other existing DSLs. Intel® XPU Backend for Triton\* is a module used by Triton to provide a reasonable tradeoff between performance and productivity on Intel® GPUs.

# SetUp Guide

Intel® XPU Backend for Triton\* serves as a backend for [OpenAI Triton](https://github.com/openai/triton). One should build from the triton repo, instead of building from intel-xpu-backend-for-triton.

## Pre-Request

Intel® XPU Backend for Triton\* requires [PyTorch](https://pytorch.org/get-started/locally/) for building, and [intel-extension-for-pytorch](https://github.com/intel/intel-extension-for-pytorch/) for running kernels on xpu.

Please follow [installation guide for intel_extension_for_pytorch](https://intel.github.io/intel-extension-for-pytorch/xpu/latest/tutorials/installation.html#installation-guide) for the detailed process and make sure the associated driver and oneAPI toolkit are installed correctly.

Note that these two should be **build from the source** for now.

## Build Intel® GPUs backend

```Bash
# Clone OpenAI/triton
git clone https://github.com/openai/triton.git
cd triton
# Clone submodules
git submodule sync && git submodule update --init --recursive --jobs 0
```
Since we are at the active development stage, it is recommended to check to latest commit for intel-xpu-backend-for-triton:

```Bash
cd third_party/intel_xpu_backend
git checkout main && git pull
```

Now Build triton with intel-xpu-backend enabled:

```Bash
cd {triton-root-dir}
cd python
TRITON_CODEGEN_INTEL_XPU_BACKEND=1 python setup.py develop
```

If you encountered any problem, please refer to the [wiki](https://github.com/intel/intel-xpu-backend-for-triton/wiki) first. This wiki also contains a [full build guide](https://github.com/intel/intel-xpu-backend-for-triton/wiki/Build-Triton-From-Scratch) for triton. It is recommended to follow that first.


# Usage Guide
Please refer to the [HowToRun.md](docs/HowToRun.md) for more information.

# Known Limitations
For known limitations, please refer to the [wiki page for known limitations](https://github.com/intel/intel-xpu-backend-for-triton/wiki/Known-Limitations).

# Contributing
It is a warm welcome for any contributions from the community, please refer to the [contribution guidelines](CONTRIBUTING.md) and [code of conduct](CODE_OF_CONDUCT.md).

## License

_MIT License_. As found in [LICENSE](https://github.com/intel/intel-xpu-backend-for-triton/blob/main/LICENSE) file.

## Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](security.md)
