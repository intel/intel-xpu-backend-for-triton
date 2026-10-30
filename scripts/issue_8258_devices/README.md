# Issue 8258: three-shape B60 / PVC comparison

The manual workflow runs exactly the three original GEMM shapes in `shapes.json`,
with B=1, BF16 inputs and FP32 output, against stock Triton and SYCL-TLA.
It records three repeats per provider, using stock autotuning and prefetch behavior.
No compiler or kernel source is changed.

The compiler baseline is 06610e5d96e1d7fb75aa7aac61e4ef61af54a4d3.
The benchmark wheel is pinned to run 40643335150 (source fd8da976), matching
our earlier B580 comparison. Kernel source identity is checked at runtime.
The default SPIR-V driver path is used; PVC retains the upstream
`PTI_COLLECTION_MODE=0` profiler workaround. Runner hardware, packages,
reference library hashes, compiler commit and selected configurations are saved.

Each shape runs in its own process with a 35-minute limit. Failures preserve
logs and partial results, and the remaining requested shapes are attempted.
Correctness verification and its existing tolerances remain enabled.
A failed shape makes the overall job fail, even if other shapes complete.

`summary.json` contains each shape's three measured times and their medians.
A positive `triton_extra_latency_pct` means Triton took longer than SYCL-TLA.
These are the existing benchmark's invocation times: SYCL-TLA includes output
allocation and initialization, while Triton reuses its output buffer.
Provider order is fixed; repeats are not independent allocations or CI runs.
These runs establish whether the relative performance gap reproduces;
they do not test the cause of a slowdown or the no-A-prefetch treatment.

CPU-only scope inspection:

```sh
python scripts/issue_8258_devices/run.py --device b60 --describe
python scripts/issue_8258_devices/run.py --device max1550 --describe
```
