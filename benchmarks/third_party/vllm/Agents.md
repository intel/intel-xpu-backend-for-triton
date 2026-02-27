# VLLM benchmarks

This folder contains scripts and utilities to run VLLM benchmarks and tests.

# VLLM installation
For the purposes of this repository, vllm installation consists of:
1. Activating a Python environment that has triton, pytorch, oneapi
2. Running [`scripts/test-triton.sh --install-vllm`](../../../scripts/test-triton.sh)

This installation internally will clone the vllm repo into a local folder, checkout the commit pinned via [`vllm-pin.txt`](vllm-pin.txt), apply our local patch [`vllm-fix.patch`](vllm-fix.patch), and apply some regexp changes to the vllm repo.

These patches are necessary because vllm doesn't yet support XPU completely.

Note that during VLLM installation we never want to install triton or pytorch, we rely on triton and pytorch from this repository which is the latest one. It is probably prebuilt for this environment. You never need to install older pytorch from VLLM requirements, we need to strip all pytorch dependencies during installation procedure.

Currently there is also an IPEX dependency in VLLM that our patches and regexps remove. We never install IPEX in our environments.

Key files for the installation procedure:
1. [`vllm-pin.txt`](vllm-pin.txt) - vllm pin that we currently use for benchmarking and testing. CI also uses this pin.
2. [`vllm-fix.patch`](vllm-fix.patch) - patch that is applied to the pin above to make tests and benchmarks run.
3. [`scripts/test-triton.sh`](../../../scripts/test-triton.sh) - script that CI and developers use to install vllm and run tests

# Environment

Usually user machine contains preistalled conda miniforge installation which you should start from.

To run benchmarks or tests you need to have triton and pytorch pre-installed. Assume that current repo folder contains python venv with preinstalled triton and pytorch from the main branch. Do not overwrite that installation with vllm dependencies.

In case existing triton installation is broken you can rebuild it with:
```
rm -rf .venv
./scripts/compile-triton.sh --venv --clean  # Compile triton
./scripts/test-triton.sh --venv # Install pytorch and do some testing
```

You can activate that env with: `source .venv/bin/activate`

# Running benchmarks
Currently there are the following benchmarks:
1. [`batched_moe`](batched_moe/) - benchmark with Batched Mixture of Experts GEMM operation.
2. `unified_attention_benchmark.py` - benchmark with unified attention. Let's forget about it for now, as it is in the process of migrating to the patch system.

Since XPU Triton requires usage of tensor descriptors, we run benchmarks two times. The first time we run the unmodified vllm version, then we patch the kernel code with tensor descriptor usage and get new performance numbers.

For the `batched_moe` benchmark there are the following files located in its folder [`batched_moe/`](batched_moe/):
1. [`batched_moe_benchmark.py`](batched_moe/batched_moe_benchmark.py) - Python script that runs the benchmark using code from vllm. As it imports the relevant kernel from vllm, it will just use whatever is available.
2. [`batched_moe.patch`](batched_moe/batched_moe.patch) - patch that we'll apply to the cloned vllm repo to add necessary changes to improve performance. Note that this patch should be generated after the general patch is applied so it should not duplicate changes from the general patch.
3. [`run_benchmark.sh`](batched_moe/run_benchmark.sh) - script that will combine the 2 steps above and run both original and modified versions of kernels.

# Running tests
We currently support only a small subset of vllm tests, as vllm requires significant changes to support XPU. The subset covers just tests for the 2 benchmarks that we have (unified attention and MOE benchmark). To run these tests, run [`scripts/test-triton.sh --vllm`](../../../scripts/test-triton.sh). It will first install vllm if necessary and then run the available tests.

# CI
There is CI for running both tests and benchmarks located in these files:
1. [`.github/workflows/third-party-tests.yml`](../../../.github/workflows/third-party-tests.yml) - CI that runs tests
2. [`.github/workflows/third-party-benchmarks.yml`](../../../.github/workflows/third-party-benchmarks.yml) - CI that runs benchmarks

Note that during the benchmarking CI there is report generation. Reports need to end with `-report.csv` and follow the format to be uploaded to the DB.

# How to update vllm pin

You can find the diff that the upstream had in a specific file by doing:
1. Go to vllm folder: `cd vllm`
2. Run `git diff main $(<../benchmarks/third_party/vllm/vllm-pin.txt) -- $FILE`

During a pin update you need to:
1. Update the pin file.
2. Ensure that the general patch is updated and applicable.
3. Ensure that [`./scripts/test-triton.sh --install-vllm`](../../../scripts/test-triton.sh) correctly installs vllm from scratch; update it if something requires changes. Keep the upstream function separate from the old one until vllm removes IPEX from dependencies.
4. Ensure that vllm tests from `test-triton.sh --vllm` run.
5. Ensure that the benchmark from the `batched_moe` folder runs before and after applying the patch from [`batched_moe.patch`](batched_moe/batched_moe.patch). Try to keep the patch minimal, for example, by keeping the same line breaks as in the upstream.
6. Update this instruction if something changed.

For all patch files, try to keep them minimal, for example, by keeping the same line breaks as in the upstream.

To install vllm you need to first remove it with `rm -rf vllm` and uninstall with `pip uninstall vllm vllm-xpu-kernels`.
