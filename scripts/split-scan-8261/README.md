# #8261 split-scan hang: reproducer, diagnostics and patch

Sources behind the numbers in #8261 and #8755. Everything runs on one Intel GPU (measured on Arc B580) with torch `b016e3e`, or with the XPU nightly `2.16.0.dev20261115+xpu`.

| File | What it is |
|---|---|
| `repro.py` | Standalone hang reproducer. Each iteration compiles `cumsum` over 50M fp32 with a fresh Inductor cache, so it re-autotunes the split-scan kernel. A faulthandler watchdog dumps stacks and exits after `--timeout` s. `--no-split` forces the non-split scan. `--ticket` is the early in-process prototype of the fix; the real fix is the patch below. |
| `run_matrix.sh` | Runs P concurrent `repro.py` processes per config: `run_matrix.sh <outdir> <iters> <timeout_s> "split:1 split:2 nosplit:3 ticket:2"`. |
| `diag2.py` | The generated split-scan kernel and Inductor's `exclusive_scan_decoupled_lookback`, copied verbatim and instrumented. It records an arrival counter as the first instruction, uses a bounded spin, and records which block a waiter gave up on and whether that block had started. The launch pattern mimics Inductor's autotune benchmark. One JSON line per burst. |
| `occupancy.py` | Measures how many work-groups of the split-scan shape are resident at once (the 80 on B580 at 16 warps). |
| `genequiv.py` | Prints the generated code for the `cumsum` case. `fakecuda` makes the patched device check see `cuda`; this is how we checked that non-XPU codegen is unchanged. |
| `split-scan-ticket-upstream.patch` | The PyTorch change (XPU-only ticket in `triton_split_scan.py`) plus the new test. Applies to PyTorch main. |
| `split-scan-ticket-test.patch` | Just the new test, `test_split_scan_block_ticket` in `test/inductor/test_torchinductor.py`. |

## Reproduce

```bash
# hang (unpatched, compute-runtime 26.35.39758.17; or 26.40 with the delay timer at 0, see below)
bash run_matrix.sh results/r1 8 240 "split:1 split:2"      # split:2 -> hung_procs=2

# driver knobs (compute-runtime debug keys)
NEOReadDebugKeys=1 ForcePreemptionMode=3 bash run_matrix.sh results/tg 8 240 "split:2"    # thread-group preemption: OK
NEOReadDebugKeys=1 ScmMidthreadPreemptionDelayTimerOverride=0 bash run_matrix.sh results/t0 8 240 "split:2"   # 26.40, timer 0: hangs

# start-order diagnostic: run 2 at once (1 for the control)
python diag2.py --bursts 20 --tag p0 & python diag2.py --bursts 20 --tag p1; wait
python occupancy.py
```

`diag2.py` fields:
- `gave_up`: blocks whose spin hit `--bound`.
- `target_not_started`: the awaited block had not executed its first instruction when the waiter gave up.
- `max_start_lag`: the furthest a block started behind its successor.

Unpatched, a lag above the resident count (`occupancy.py`) means the deadlock is possible.

`diag2.py` hard-codes the 50M-element `cumsum` shape (`R0_BLOCK=2048`, 16 warps) and the workspace size Inductor allocates for it at `b016e3e`.
