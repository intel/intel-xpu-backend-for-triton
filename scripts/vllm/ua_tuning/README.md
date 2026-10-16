# Unified-attention CI tuning

These Intel CI helpers call a temporarily patched copy of the shared
[model-based tuner](../../../benchmarks/triton_kernels_benchmark/vllm/unified_attention/tune_unified_attention.py).
Apply the unified-attention patch to the pinned vLLM installation first, using
an environment with the matching XPU PyTorch and Triton builds.
The helpers support Intel XPUs with the required tensor-descriptor and graph
APIs. Pass `--device xpu:N` to both `shard.py` and `tune_ci.py` to select a GPU
(default: `xpu:0`). Workload memory filtering uses that GPU, and exported profiles
use its detected device name. Retune on each device model; existing profiles
only apply when their device identity matches.

Run from the repository root:

```bash
export PYTHONPATH="$PWD/benchmarks:$PWD/vllm${PYTHONPATH:+:$PYTHONPATH}"
python scripts/vllm/ua_tuning/shard.py --output reports/manifests
python scripts/vllm/ua_tuning/tune_ci.py --tune --clear-cache \
  --manifest reports/manifests/bfloat16.json \
  --save-dir reports/configs --measurements reports/bf16-measurements.json
python scripts/vllm/ua_tuning/tune_ci.py --tune --clear-cache \
  --manifest reports/manifests/float8_e4m3fn.json \
  --save-dir reports/configs --measurements reports/fp8-measurements.json
```

Omit `--tune` to benchmark existing profiles. `--measurements` applies only to
tuning and saves progress records; automatic resume is not implemented.

- `tune_ci.py` validates workload manifests and reuses the CI benchmark's input
  construction, including its fixed seed, contiguous tensors, FP8 scales and
  scratch buffers. It copies the base tuner into a temporary directory, checks
  and applies `tuner-ci.patch`, then loads that copy. The repository copy stays
  upstreamable and unchanged.
- `timing.py` provides the optional cache-cleared graph timer. `--clear-cache`
  evicts 256 MB before each replay and excludes eviction from attention timing.
- `shard.py` imports the benchmark's `ATTENTION_CONFIGS_BF16` and
  `ATTENTION_CONFIGS_FP8` lists and converts them to tuning workloads. It sets
  `FP8=0` and `DEBUG_BENCH=0` before importing so neither list is truncated or
  restricted by the caller's benchmark shortcuts. The benchmark's normal dtype
  and device-memory filters still apply. Importing requires a supported GPU,
  vLLM and the benchmark dependencies (`pandas<3.0`, scipy, psutil, tabulate,
  matplotlib).

Generated workload snapshots and their partition assignments are saved in
`reports/manifests/`. Pass a saved manifest to `tune_ci.py` to replay those inputs.
Each entry contains the benchmark arguments `q_heads`, `k_heads`, `head_size`,
`qdtype`, `seq_lens`, `sliding_window`, `soft_cap`, `num_blocks` and `block_size`,
plus an optional unique `id`. `qdtype` is `null` for BF16 or `"float8_e4m3fn"`
for FP8; `seq_lens` contains `[Q, KV]` pairs. Windows and soft caps use `null`
when disabled. Layouts, output dtype and scratch settings come from the benchmark.
Regenerate older manifests that used the synthetic tuner's schema.

An empty manifest (`[]`) prints a skip message and exits successfully before
loading PyTorch, vLLM or the tuner. This handles parts without inputs for a dtype.

Input allocation and timing are passed explicitly to the patched tuner. Candidate
search, confirmation, winner selection and profile export run in that copy.

Check patch compatibility without importing PyTorch or vLLM:

```bash
python scripts/vllm/ua_tuning/tune_ci.py --check-patch
```

Refresh `tuner-ci.patch` against the base tuner when its implementation changes.

## Branch and workflow

`ua-offline-tuning-ci` extends `ua-offline-tuning` with these helpers and CI
profile loading. `tuner-ci.patch` adds
measurement checkpoints, finalist confirmation and allocator/timer hooks for CI.
The shared tuner, kernel patch and profile JSONs are inherited unchanged from
the base branch.

The manual workflow in `.github/workflows/ua-offline-tuning.yml` installs the
XPU environment, applies the UA patch and tunes BF16 followed by FP8. It uses
cache-cleared graph timing and uploads profiles, measurement checkpoints and
logs. It does not run the experimental timing diagnostic or publish configs.
Set `runner_label` to the target Intel XPU runner (default: `b580`) and `device`
to its PyTorch device (default: `xpu:0`). The workflow uses the same device for
manifest generation and both tuning steps. As in the vLLM test workflows,
`max*` runners use the rolling driver label and `lts` selects the LTS PyTorch
build. The environment and candidate pool currently target XPU; supported
workloads still depend on the device's memory and dtype capabilities.

The workflow defaults to `all`, scheduling four parts sequentially as separate
jobs; each job can use a different runner with the selected label. Select `1`,
`2`, `3` or `4` to run only one part. Workflow runs share a concurrency group
so separate tuning runs cannot overlap.
`shard.py --part N` selects one of the four parts. A stable hash of each profile
key determines its part, keeping each profile file within one part even when
runners filter out different workloads. Part sizes can differ, and the tuner
skips empty dtype manifests. The commands above select all inputs without
splitting them. Four-part assignments replace
the previous two-part assignments; regenerate manifests before a new run.

GitHub requires a dispatch workflow to exist on the default branch before it
can be launched through the normal manual-dispatch interface.
