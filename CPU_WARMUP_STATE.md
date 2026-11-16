# Compile-cache warmup on CPU runners: current state

Written 2026-11-15. Branch `amyachev/cpu-runner-compile-warmup`, draft PR #8805.

## Goal

Warm the Triton compile cache once on a GPU-less CPU runner, so the GPU test shards restore it
instead of compiling. Today every `integration-tests` shard runs `warm-compile-cache` on a GPU runner,
which holds the GPU while doing CPU-bound work.

## How it works

1. `TRITON_INTEL_TARGET_JSON` (knob in `python/triton/knobs.py`, offline mode in
   `third_party/intel/backend/driver.py`): the Intel driver reports a device recorded in a JSON file
   instead of querying hardware. `scripts/dump_xpu_target.py OUT.json` writes it (target, device
   properties, `ocloc --version`); `--check FILE` compares `ocloc --version` with the GPU runner's.
2. `build-test-reusable.yml`, input `warmup_on_cpu` (default false; `build-test.yml` sets it to
   `inputs.runner_label == ''`):
   - `build` job dumps the target on the GPU runner and uploads it (`xpu-target-*`).
   - New `warmup-cache` job on a CPU runner: installs the wheel, "Report environment details", sets
     `TRITON_INTEL_TARGET_JSON`, "Check the driver stack matches the GPU runner", runs the seven
     `warm-compile-cache` steps with `capture-workers: 4` (pytest-xdist), uploads `cache/`, `trace/` and
     `cache_dir` as artifact `triton-compile-cache-*` (retention 1 day).
   - `integration-tests` waits for it, downloads the artifact into `triton.knobs.cache.dir`, rewrites
     `cache_dir` in the trace `*.jsonl` (the completeness report pairs records by `(cache_dir, hash)`), and skips
     its own warm steps (`!inputs.warmup_on_cpu`). `check-jobs` allows `warmup-cache` to be skipped.
3. Plugin `python/triton/_compile_warmup.py` (FakeTensorMode based): a test counts as warmed only if it
   produced warmup records, and completeness is checked only for those tests. A test skipped before any
   launch is therefore simply not required to be warm.

## Runner labels (the open infra problem)

- CPU runners are labelled `cpu` and `runner-0.0.22`; `lts-cpu` is also a `cpu` runner, so `cpu`
  overlaps with it. `runs-on` labels cannot exclude a runner.
- The `cpu` runner that picked up a rolling job (`triton-idc-4-5-cpu`, `triton-idc-4-6-cpu`) had an old
  stack (`AGAMA_VERSION=1146`, `LIBIGC1_VERSION=2.11.43-1146`, `ocloc` without `--version`). Its `ocloc`
  rejected the SPIR-V: `unknown extension 'SPV_INTEL_predicated_io'` (190 `test_cast_matmul` compiles).
  The compiler enables predicated IO from the driver version only (`has_predicated_io = not is_lts`,
  `compiler.py:323`), nothing probes the local `ocloc`. `PREDICATED_IO_PROBE` in `scripts/pytest-utils.sh`
  checks `spirv-dis`, a different tool, and is not used by the warmup job.
- Current `runs-on` of `warmup-cache`: `lts` -> `["linux", "lts-cpu", runner_version]`;
  rolling -> `["cpu", "name:triton-idc-3"]` (temporary pin, FIXME: use a label for rolling CPU runners).
- The check script calls `ocloc --version`; an old `ocloc` has no such flag and the script crashes with a
  traceback (exit 226) instead of reporting a mismatch. Not fixed. Idea: compare the `capture-hw-details.sh`
  versions (`LIBIGC1_VERSION`, `AGAMA_VERSION`) instead, since they work on old stacks.

## Commits on the branch (all pushed)

```
81db13d021 Pin the rolling warmup job to a named CPU runner
372ddfe658 Pick the warmup CPU runner by driver: lts-cpu for lts, cpu otherwise
cee89c8b8a Select the warmup CPU runner by driver version and check the driver stack
fff9635df5 Warmup plugin: handle more torch calls that need a device (__setitem__, copy_, .to())
37ccab0149 Make the compile warmup work without a GPU (torch.tensor, __getitem__, .contiguous(), capture-workers)
b825d6160f Warm the Triton compile cache on a CPU runner
```

## Uncommitted (python/triton/_compile_warmup.py, +27 lines)

Active only when `TRITON_INTEL_TARGET_JSON` is set:
- `torch.xpu.get_device_capability` / `get_device_properties` answer from the target (fixes
  `RuntimeError: No XPU devices are available` in `opt_flags_intel.py:31 compute_split_k`; the emulated
  `gpu_subslice_count` is 56, same as the real GPU).
- `torch.xpu.device` is a no-op context manager (`fpsan.py` `_launch`).
- Autograd is emulated instead of skipped (building the real graph on an xpu fake tensor segfaults without a
  device): `requires_grad` only flags the tensor, `torch.autograd.Function.apply` is replaced by a version that
  calls `forward` with a stand-in ctx and remembers `backward`, `Tensor.backward` calls it, and `.grad` returns
  an empty fake tensor for flagged tensors. Verified offline: test_reduce/test_topk/test_swiglu, 1200 passed,
  0 autograd skips, `_topk_backward` and `_reduce_backward` are captured. Real-GPU run against that cache: 1302 passed, 1069 hits, 34 misses, all in tests not marked `enable_warmup` (`test_reduce_mxfp_scale_boundaries`, `test_swiglu.py::test_op[*-1311-4352]`), so every warmed autograd test hits.


## Why torch calls crash without a device

FakeTensorMode works at the dispatcher. Anything that takes a C++ device guard before dispatch needs a real
device: `torch.tensor(data, device=...)`, `Tensor.__getitem__`, `__setitem__`, `copy_`, `.to(<xpu>)`,
`.contiguous()` on a non-contiguous tensor, `torch.xpu.device(...)`, and any autograd graph on an xpu tensor.
`ZE_ENABLE_NULL_DRIVER=1` does not help (the null driver is a loader test stub; torch gets no device).
`new_tensor(data)` still segfaults; nothing in the suites run so far uses it.

## Results (local, Max 1100 with the GPU hidden by `ZE_AFFINITY_MASK=99`)

| suite | offline warmup | real GPU run against that cache |
|---|---|---|
| regression | 696 passed, rc 0 | 696/696 hits |
| language (full warmup target) | 6415 passed; only 17 `test_sort` M=N=1 failures, fixed after | affected tests 948/948 hits |
| triton_kernels | 2880 passed, 0 failed, rc 0 (CI had 152 failed) | not finished, see below |

triton_kernels real run (killed at ~98%, partial trace): 992 misses, of which 865 are in tests with no
warmup record (the skipped autograd tests `test_reduce.py::test_op`, `test_topk_arbitrary_k_forward_backward`,
and tests not marked `enable_warmup`), which is expected. 127 misses are inside warmed tests and are NOT
explained yet: `_empty_fn` (60) and `_fpsan_kernel` (12) in `test_matmul_mixed_fp8_matches_upcast`,
`_topk_forward` (48) and a few others in `test_topk_fpsan_masks_padded_experts`. For `_topk_forward`,
`_empty_fn`, `_fpsan_kernel` the real hash equals the warmup hash, so the key matches; a kernel that is
compiled but never stored (compile failure or resource probe) would look like a miss every time.
The GPU-visible baseline for those two tests (`/tmp/spike/tk_base.sh`) was killed before finishing; it would
show whether these misses are inherent to the tests.

## Related PRs / issues

- #8805 (this work, draft).
- #8826 (draft, branch `amyachev/arch-parser-no-dlclose`): `ArchParser.__del__` looked `dlclose` up through a
  library handle with no dependencies, so it always failed ("Exception ignored in ArchParser.__del__" in
  every warmup worker log). Fix takes `dlclose` from libc and keeps the parser alive in the function it hands
  out. `intel/test_spirv_utils_no_unload_7682.py` expects `ArchParser` to keep a destructor.
- #8804 (draft, by the user) and issue #8808: warmup for `python/test/unit/intel`.
- Review comment on #8805 (kwasd): the driver stack must match, call the CPU runner with the driver label.

## Next steps

1. Run `pre_commit` on `_compile_warmup.py`, commit and push the uncommitted plugin changes.
2. Rerun the GPU-visible baseline for `test_matmul_mixed_fp8_matches_upcast` and
   `test_topk_fpsan_masks_padded_experts`, then decide how to treat the 127 misses (exclude those tests from
   warmup, or fix what makes them miss).
3. Run the remaining suites offline: matmul, scaled_dot, gluon, tutorials (Flash Attention).
4. Get a rolling CPU runner label, replace `name:triton-idc-3`; make `dump_xpu_target.py --check` robust on
   an old `ocloc`.
5. Measure artifact size and total warmup time in CI (`Collect compile cache` prints `du`).

## Reproduce locally

```bash
# 1. record the target on the GPU box
python scripts/dump_xpu_target.py /tmp/spike/t2.json
# 2. warm with the GPU hidden (skiplist args as in the action; run from python/triton_kernels)
ZE_AFFINITY_MASK=99 TRITON_INTEL_TARGET_JSON=/tmp/spike/t2.json TRITON_XPU_GEN_NATIVE_CODE=1 \
  TRITON_CACHE_DIR=/tmp/c TRITON_CI_COMPILE_TRACE_DIR=/tmp/tr TRITON_CI_CACHE_PHASE=warmup-triton_kernels \
  pytest -p triton._compile_warmup -s --tb=line --warmup-only -n 4 --warmup-workers 32 --device xpu \
  --xfail-norun-from-file=scripts/skiplist/default/triton_kernels.xfail.txt --select-fail-on-missing \
  tests -k 'not test_mxfp' -q
# 3. real run on the GPU with the SAME TRITON_CACHE_DIR and TRITON_CI_CACHE_PHASE=triton_kernels, then
python -m triton._test_runner report --directory /tmp/tr [--require-complete]
```
Helper scripts used in this session are in `/tmp/spike/` (`tk.sh`, `tk3.sh`, `tk_base.sh`, `lang.sh`, ...),
the dumped target is `/tmp/spike/t2.json`. `pytest-skip` must be installed
(`pip install -r scripts/requirements-test.txt`) for `--xfail-norun-from-file`.
Warmup leaves orphaned `multiprocessing.spawn` pool workers behind (1368 were found, mostly idle).
