# VLLM benchmarks

This folder contains scripts and utilities to run VLLM benchmarks and tests.

# VLLM installation
For purposes of this repository vllm installation consists of:
1. Activating python environment that has triton, pytorch, oneapi
2. Running `$REPO/scripts/test-triton.sh --install-vllm`

This installation internally will clone the vllm repo into a local folder, checkout commit pinned via `vllm-pin.txt` file, apply our local patch `vllm-fix.patch` and apply some regexp changes to vllm repo.

These patches are necessary because vllm doen't yet support XPU completely.

Currently there is also IPEX dependency in VLLM that our patches and regexps remove. We never install IPEX in our environments.

Key files for installation procedure:
1. `benchmarks/third_party/vllm/vllm-pin.txt` - vllm pin that we currently use for benchmarking and testing. CI also uses this pin.
2. `benchmarks/third_party/vllm/vllm-fix.patch` - patch that is applied to the pin above to make tests and benchmarks run.
3. `scripts/test-triton.sh` - script that CI and developers use to install vllm and run tests

# Running benchmarks
Currently there are following benchmarks:
1. `batched_moe` - benchmark with Batched Mixture of Experts GEMM operation.
2. `unified_attention_benchmark.py` - benchmark with unified attention. Let's forget about it for now, as it is in the process of migrating to patch system.

Since XPU triton requires usage of tensor descriptors we run benchmarks two times. First time we run unmodified vllm version, then we patch the kernel code with tensor descriptor usage and get new perfomance numbers.

For `batched_moe` benchmark there are following files located in it's folder `benchmarks/third_party/vllm/batched_moe`:
1. `batched_moe_benchmark.py` - python script that runs the benchmark using code from vllm. As it imports the relevant kernel from vllm it will just use whatever is available.
2. `batched_moe.patch` - patch that we'll apply to cloned vllm repo to add necessary changes to improve performance. Note that this patch should be generated after the general patch is applied.
3. `run_benchmark.sh` - script that will combine 2 steps above and run both original and modified version of kernels.

# Running tests
We currently support only a small subset of vllm tests, as vllm requires significant changes to support XPU. The subset covers just tests for 2 benchmarks that we have (unified attention and MOE benchmark). To run these tests run `scripts/test-triton --vllm`, it will first install vllm if necessary and then run available tests.

# CI
There is CI for running both tests and benchmarks located in these files:
1. `.github/workflows/third-party-tests.yml` - CI that runs tests
2. `.github/workflows/third-party-benchmarks.yml` - CI that runs benchmarks

Note that during the benchmarking CI there is report generations. Reports need to end with `-report.csv` and follow the formant to be uploaded to the DB.

# How to update vllm pin

You can find the diff that the upstream had in specific file by doing:
1. Go to vllm folder `cd vllm`
2. Run `git diff main $(<../benchmarks/third_party/vllm/vllm-pin.txt) -- $FILE

During pin update you need to
1. Ensure that patch is updated and applicable.
2. Ensure that benchmark from `batched_moe` folder runs before and after applying the patch from `batched_moe.patch`
3. Ensure that `./scripts/test-triton --install-vllm` correctly installs vllm from scratch, update it if something requires changes. Keep upstream function separate from old one until vllm removes IPEX from depencies.
4. Ensure that vllm tests from `triton-test.sh --vllm` run
5. Update the pin file
6. Update this instruction if something changed

To install vllm you need to first remove it with `rm -rf vllm` and uninstall with `pip uninstall vllm vllm-xpu-kernels`
