# Unified-attention CI tuning

These Intel CI helpers call the shared
[model-based tuner](../../../benchmarks/triton_kernels_benchmark/vllm/unified_attention/tune_unified_attention.py).
Apply the unified-attention patch to the pinned vLLM installation first, using
an environment with the matching XPU PyTorch and Triton builds.

Run from the repository root:

```bash
export PYTHONPATH="$PWD/vllm${PYTHONPATH:+:$PYTHONPATH}"
python scripts/vllm/ua_tuning/tune_ci.py --tune --clear-cache \
  --manifest scripts/vllm/ua_tuning/inputs/bf16.json \
  --save-dir reports/configs --measurements reports/bf16-measurements.json
python scripts/vllm/ua_tuning/tune_ci.py --tune --clear-cache \
  --manifest scripts/vllm/ua_tuning/inputs/fp8.json \
  --save-dir reports/configs --measurements reports/fp8-measurements.json
```

Omit `--tune` to benchmark existing profiles. `--measurements` applies only to
tuning and saves progress records; automatic resume is not implemented.

- `tune_ci.py` reuses the CI benchmark's input construction, including its fixed
  seed, contiguous tensors, FP8 scales and scratch buffers. Each workload must
  include `num_blocks`.
- `manifest.py` parses custom workloads and calls the shared tuner. Running it
  directly uses synthetic initialization rather than the exact CI fixture.
- `timing.py` provides the optional cache-cleared graph timer. `--clear-cache`
  evicts 256 MB before each replay and excludes eviction from attention timing.
- `inputs/` contains the BF16 and FP8 manifests used by the completed CI tuning
  run. They are saved workload lists, not automatically regenerated from the
  benchmark's current configuration list.

Input allocation and timing are passed explicitly to the shared tuner. Candidate
search, confirmation, winner selection and profile export remain in that tuner.
The shared tuner does not import these CI helpers.

## Branch and workflow

`ua-offline-tuning-ci` extends `ua-offline-tuning` with these helpers, CI profile
loading and the temporary SYCL benchmark skip. The shared tuner, kernel patch
and profile JSONs are inherited unchanged from the base branch.

The manual workflow in `.github/workflows/ua-offline-tuning.yml` installs the
XPU environment, applies the UA patch and tunes BF16 followed by FP8. It uses
cache-cleared graph timing and uploads profiles, measurement checkpoints and
logs. It does not run the experimental timing diagnostic or publish configs.

The workflow defaults to both parts, scheduled sequentially as separate jobs;
each job can use a different B580 runner. Select `1` or `2` to run only one part.
`shard.py` reads `inputs/parts.json` and keeps each profile file within one part.
The commands above tune the complete manifests without splitting them.

GitHub requires a dispatch workflow to exist on the default branch before it
can be launched through the normal manual-dispatch interface.
