# Developer Guide

This short guide covers creating a basic development environment, installing needed dependencies, and best practices for building and running Triton on different Intel platforms.

## Dependencies

The primary software dependency is oneAPI. The best way to get oneAPI for PyTorch / Triton development is to use the [Intel PyTorch Dependency Bundle](https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpus.html). Scroll down to "Step 2: Install Intel Support Packages" and choose an appropriate option for your system.
* **Option 2A-2C: Install using System Package Manager** is recommended for system-wide installs of the currently released packages with automatic updates.
* **Option 2D: Offline Install** is recommended for more control over package versions, or to have multiple installations of different versions of the oneAPI dependency bundle on the same machine.

## GPU Drivers

Intel GPU drivers and development packages are required. Several release streams are available. Ubuntu 22.04 release stream information is linked below. Other release stream information is available at the same link.

* [Ubuntu LTS Graphics Driver](https://dgpu-docs.intel.com/driver/release-streams.html#ubuntu-long-term-support-lts-recommended)
* [Ubuntu Production Graphics Driver](https://dgpu-docs.intel.com/driver/release-streams.html#ubuntu-production)
* [Ubuntu Rolling Stable Graphics Driver](https://dgpu-docs.intel.com/driver/release-streams.html#ubuntu-rolling-stable)


### Data Center GPU Max Series (Ponte Vecchio)

The LTS, Production, and Rolling Stable release streams are supported. Some performance optimizations may not be available with the LTS driver package. For development, we recommend rolling stable.

### Client GPUs

The Rolling Stable release stream is supported. Other release streams may not work for all test cases.

## Tips and Tricks

* Building Triton with `compile-triton.sh` can be memory intensive due to compiling several large dependencies from source. Use the environment variable `MAX_JOBS=8` on machines with < 64GB of memory.
* Pre-built wheels will be fetched during invocation of the `test-triton.sh` script if the GitHub CLI tool `gh` is installed. The default version on Ubuntu 22.04 is too old to support some flags we use, so install the latest from GitHub APT repo using [these instructions](https://github.com/cli/cli/blob/trunk/docs/install_linux.md).
