# Explicit GRF comparison

This test branch runs the existing four-part B580 BF16/FP8 workload manifests
and offline candidate grid from `ua-offline-tuning-weekly` at `701dbc0a1a`.
Only the requested GRF modes change: `default` becomes explicit `128`, and
`256` remains available. Existing legality pruning is preserved.

`compare_grf.py` checks every launchable candidate against a chunked PyTorch
paged-attention reference, using the benchmark's BF16/FP8 tolerances. It checks
`grf_count` in the final native binary's `.ze_info` for the attention kernel;
auxiliary reduction kernels are recorded separately. Resource-limit failures
are excluded. Correctness failures and unexpected native GRF modes fail the job.

Each workload selects the lowest three-round median separately for each mode.
The two winning configurations may differ in any other tuning parameter.
Five alternating timing pairs then compare those frozen configurations using
the existing cold graph timer. A winner needs more than a 3% latency reduction
relative to the slower mode. Apparent 128 wins receive a fresh-process check
with the same inputs and frozen configurations; a non-confirmed win is labeled
`unconfirmed_128` and excluded from the confirmed 128 count.

Artifacts contain `comparison.csv`, per-dtype `summary.json`, full candidate
measurements and native GRF/spill metadata in each workload's `search.json`,
and the winning native binaries. The environment and complete input manifests
are recorded by the workflow. No minimum number of 128 wins is required.
Results apply to the finite existing grid and the workloads that fit the
runner's benchmark memory filter. They do not establish a global optimum or
which single shared configuration would best serve every workload with a key.

The workflow is manually dispatched and report-only, with no config PR job.
Local CPU checks:

```sh
python scripts/vllm/ua_tuning/tune_ci.py --check-patch
pytest -s --tb=short scripts/vllm/ua_tuning/test_compare_grf.py
```
