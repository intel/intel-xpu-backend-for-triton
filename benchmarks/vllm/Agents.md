# VLLM benchmarks

This folder contains scripts and utilities to run VLLM benchmarks and tests.

Note, if you just want to update the pin, try this prompt: `Update vllm pin to COMMIT_HASH` and make sure that agent have this file in the context.

# VLLM installation
For the purposes of this repository, vllm installation consists of:
1. Activating a Python environment that has triton, pytorch, oneapi
2. Running [`scripts/test-triton.sh --install-vllm`](../../scripts/test-triton.sh)

This installation internally will clone the vllm repo into a local folder, checkout the commit pinned via [`vllm-pin.txt`](vllm-pin.txt), apply our local patch [`vllm-fix.patch`](vllm-fix.patch), and apply some regexp changes to the vllm repo.

These patches are necessary because vllm doesn't yet support XPU completely.

Note that during VLLM installation we never want to install triton or pytorch, we rely on triton and pytorch from this repository which is the latest one. It is probably prebuilt for this environment. You never need to install older pytorch from VLLM requirements, we need to strip all pytorch dependencies during installation procedure.

The new pin no longer has an IPEX dependency. We never install IPEX in our environments.

Key files for the installation procedure:
1. [`vllm-pin.txt`](vllm-pin.txt) - vllm pin that we currently use for benchmarking and testing. CI also uses this pin.
2. [`vllm-fix.patch`](vllm-fix.patch) - patch that is applied to the pin above to make tests and benchmarks run.
3. [`scripts/test-triton.sh`](../../scripts/test-triton.sh) - script that CI and developers use to install vllm and run tests

# Environment

Usually user machine contains preistalled conda miniforge installation which you should start from. Make sure to activate oneapi though with `source /opt/intel/oneapi/setvars.sh --force `

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
2. [`unified_attention`](unified_attention/) - benchmark with unified attention.

Since XPU Triton requires usage of tensor descriptors, we run benchmarks two times. The first time we run the unmodified vllm version, then we patch the kernel code with tensor descriptor usage and get new performance numbers.

Both benchmarks follow the same structure. For each benchmark folder (e.g. [`batched_moe/`](batched_moe/), [`unified_attention/`](unified_attention/)):
1. `<name>.patch` - patch that we'll apply to the cloned vllm repo to add necessary changes to improve performance. Note that this patch should be generated after the general patch is applied so it should not duplicate changes from the general patch.

The benchmark Python scripts for both benchmarks are located in [`../triton_kernels_benchmark/vllm_<name>_benchmark.py`](../triton_kernels_benchmark/) and integrated with the `triton_kernels_benchmark` package.

The shared [`run_benchmark.sh`](run_benchmark.sh) script orchestrates both steps. It takes the benchmark folder name as the first argument and derives the patch file and benchmark script from the `NAME` convention above. It applies the pattern: run without patch (`TD_PATCHED=0`), apply patch, run again (`TD_PATCHED=1`), revert patch. Any extra arguments (e.g. `--reports`) are forwarded to the benchmark script.

You can run a benchmark with environment variable `DEBUG_BENCH=1` to speed up if the benchmark runs at all. For example:
```
DEBUG_BENCH=1 bash benchmarks/vllm/run_benchmark.sh unified_attention
```

# Running tests
We currently support only a small subset of vllm tests, as vllm requires significant changes to support XPU. The subset covers just tests for the 2 benchmarks that we have (unified attention and MOE benchmark). To run these tests, run [`scripts/test-triton.sh --vllm`](../../scripts/test-triton.sh). It will first install vllm if necessary and then run the available tests.

# CI
There is CI for running both tests and benchmarks located in these files:
1. [`.github/workflows/vllm-tests.yml`](../../.github/workflows/vllm-tests.yml) - CI that runs tests
2. [`.github/workflows/triton-benchmarks-pvc.yml`](../../.github/workflows/triton-benchmarks-pvc.yml) - CI that runs benchmarks
3. [`.github/workflows/triton-benchmarks-bmg.yml`](../../.github/workflows/triton-benchmarks-bmg.yml) - CI that runs benchmarks on BMG

All vLLM benchmarks run via `triton-benchmarks.yml` with conditional execution:

```bash
# Run specific vLLM benchmarks
gh workflow run triton-benchmarks.yml --field benchmarks='["vllm_unified_attention_benchmark.py", "vllm_batched_moe_benchmark.py"]'

# Skip vLLM benchmarks
gh workflow run triton-benchmarks.yml --field skip_benchmarks='["vllm_unified_attention_benchmark.py", "vllm_batched_moe_benchmark.py"]'
```

Note that during the benchmarking CI there is report generation. Reports need to end with `-report.csv` and follow the format to be uploaded to the DB.

# How to update vllm pin

You can find the diff that the upstream had in a specific file by doing:
1. Go to vllm folder: `cd vllm`
2. Run `git diff main $(<../benchmarks/vllm/vllm-pin.txt) -- $FILE`

During a pin update you need to:
1. Update the pin file.
2. Ensure that the general patch is updated and applicable.
3. Ensure that [`./scripts/test-triton.sh --install-vllm`](../../scripts/test-triton.sh) correctly installs vllm from scratch; update it if something requires changes.
4. Ensure that vllm tests from `test-triton.sh --vllm` run.
5. Ensure that the benchmarks for `batched_moe` and `unified_attention` folders work as expected. That includes checking that appropriate patch can be applied wihout any issues. Try to keep the patch minimal, for example, by keeping the same line breaks as in the upstream. You can use `DEBUG_BENCH=1` env variable to test if benchmark runs.
6. Update this instruction if something changed.

To install vllm you need to first remove it with `rm -rf vllm` and uninstall with `pip uninstall vllm`.

# How to update patch

When updating a patch file it is easy to make mistake or get a corrupt patch. If you have a corrupt patch don't try to fix it in-place, it's too easy to make mistakes that way. Instead try to apply what you have (maybe partially) and then reproduce the desired file state in a local repo clone. Then generate proper patch file from the local repo.

For all patch files, try to keep them minimal, for example, by keeping the same line breaks as in the upstream.

Sometimes you might notice that patch solved some problem for IPEX compatibility that is no loger in the source code. Try to get rid of that part of the patch then and check if tests and benchmarks work that way.

Actively use env variable `DEBUG_BENCH=1` to check if benchmark runs at all. It is much faster than watiting for all configurations.
