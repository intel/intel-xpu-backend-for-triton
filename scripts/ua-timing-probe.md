# XPU graph/event timing diagnostic

Timing investigation only; no tuning profiles are generated. The manual `timing test` workflow runs this diagnostic on a B580 runner. The script itself never dispatches CI or changes installed packages.

## Purpose

Reproduce the anomalous event timings from runner `sdp-123088` without a full tuning sweep. Prior failures included both inflated short decode times and severely underestimated long prefill/mixed times. The same probes should eventually run on that runner and a working runner; local success alone cannot establish that CI is fixed.

`probe.py` runs each experimental condition in a **fresh subprocess**, sequentially. The default matrix has 48 processes:

- BF16 matrix multiplication, CI BF16 decode `0077`, CI FP8 prefill `0087`.
- Eager calls or graph replay.
- No eviction or a 256 MiB zero-fill before each call.
- Fresh events around each call or fresh events around the whole batch.
- No profiler anywhere in that process, or a profiler session before each timing round.

Optional workloads: `matmul_long` (64 matmuls per invocation), `ua_mixed` (`0091`, the case with approximately 0.26 ms event time versus 13.46 ms profiler time).

Each condition warms/compiles before measurement, captures when requested, checks replay output against eager output, and records three timing rounds by default. The output check detects replay discrepancies; it is not an independent attention correctness test. A trial repeats the condition in another fresh process.

For attention, the probe uses the existing CI fixture and a **fixed main configuration**: BLOCK_M=32 for decode and 64 for prefill/mixed. `--block-m` overrides this. Main's candidate pool is restricted to that single config before warmup; no online search occurs inside timing. The input fixture preserves the CI cache allocations, dtype and layout.

## Commands

Run from the repository root with an existing XPU Torch/Triton/vLLM environment and oneAPI activated. Do not change Torch/compiler versions in place. Use separate existing environments for version comparisons.

Preview the process matrix without importing GPU packages or allocating a GPU:

```bash
python scripts/ua_timing_probe.py \
  --repo . --output /tmp/ua-probe-preview --dry-run
```

Small matrix-multiplication smoke test (two processes, eager and graph; no profiling/eviction):

```bash
python scripts/ua_timing_probe.py \
  --repo . --output /tmp/ua-timing-smoke \
  --workload matmul_short --eviction off --profiler none \
  --events per_replay --rounds 2 --repeats 4
```

Full default matrix, using the already prepared main TD source from the prior local diagnostic:

```bash
python scripts/ua_timing_probe.py \
  --repo . \
  --ua-source tmp/ua-cold-timing-investigation-20261001/ua-local/main-source \
  --output /tmp/ua-timing-matrix
```

`--ua-source` must point to a **main TD-patched vLLM source root**, containing `vllm/v1/attention/ops`. It is not the unpatched pinned source and not the offline-tuned implementation. The installed vLLM environment must be compatible. The script reads these sources and does not modify them. On another machine, supply its corresponding prepared main source. No source checkout or patch application is automatic.

To repeat only graph-related conditions on long synthetic and mixed attention workloads:

```bash
python scripts/ua_timing_probe.py \
  --repo . \
  --ua-source tmp/ua-cold-timing-investigation-20261001/ua-local/main-source \
  --output /tmp/ua-timing-long --workload matmul_long ua_mixed \
  --execution graph --trials 2
```

Use a new output directory each time. A worker timeout (default ten minutes) stops the matrix. Completed worker JSONs/logs remain available. This tool does not automatically retry failed or suspicious cases.

For a hardware/software record alongside results, run the existing `scripts/capture-hw-details.sh` and `python -m pip freeze` separately into report files. The probe also saves Torch build information, device properties, selected runtime environment variables, attention source hash, and fixed launch config. Do not run other workloads on the same GPU during the diagnostic.

## Reading the results

- `plan.json`: exact matrix and measurement settings.
- `status.json`: completed processes and any failure.
- `summary.csv`: round median/range, event time, host time, and independent host time without timing events.
- Numbered JSONs: workload/config, environment, all per-call event samples and per-round measurements.
- Numbered logs: compilation/runtime/profiler warnings.
- Trace JSONs: profiler-session kernel and CPU events, for conditions that request profiling.

**Timing scopes differ deliberately:**

| Measurement | Includes eviction? | Includes host submission overhead? |
|---|---|---|
| Per-call events | No | May include device idle gaps inside the event interval |
| Batch events | Yes, when enabled | May include device idle gaps |
| Synchronized host wall time | Yes, when enabled | Yes |
| Independent wall reference, no timing events | Yes, when enabled | Yes |

Do not interpret a larger cold batch time as a regression against eviction-excluding per-call events. Start with **eviction disabled**, where the measured work matches more closely. For long workloads, large event undercounts relative to both wall references are suspicious. For short workloads, host overhead can dominate, so wall/event differences alone are not proof of a timestamp defect.

Compare otherwise identical conditions:

1. Eager stable, graph unstable: investigate the graph/event interaction.
2. Only profiler conditions unstable: investigate profiler interference/lifecycle.
3. Only eviction conditions unstable: investigate the two-graph submission pattern.
4. Per-call events unstable, batch events stable: investigate timing boundaries/event ordering.
5. Both wall time and event time unstable: investigate real scheduling, contention, or execution variation too.

Profiler traces are diagnostic evidence, not an unquestioned ground truth: the failing CI logs themselves include PTI timestamp warnings. Do not use these measurements to change tuning winners.

## Local validation

Six CPU tests check timer scope, fresh-event allocation, no-event reference behavior, matrix generation without Torch, and argument validation. Python syntax compilation also passes. Six local GPU smoke conditions completed: eager/graph decode and matmul, plus cold graph FP8 prefill/mixed with profiling. These check execution and output consistency, not timing stability; the short smoke samples include variable event overhead. The full failing-runner matrix still needs CI validation. The workflow tests all five workloads with two fresh-process trials per condition (160 processes).
