# Compiler default versus forced 256 GRFs

This experiment asks whether explicitly setting `grf_mode="256"` improves
unified-attention performance over letting the compiler select the GRF mode.
It uses the same four-part B580 BF16/FP8 manifests and offline candidate grid
as the earlier 128-versus-256 experiment at `aa2862a24f` (the underlying tuning
base is `ua-offline-tuning-weekly` at `701dbc0a1a`). No kernel or compiler code
changes are included.

The two arms are `default` and `256`. The default arm stores `grf_mode=None`
in `AttentionConfig`, so the launch omits the compiler option entirely.
The script verifies that omission. It reads the final native `.ze_info`
GRF/spill counts for both arms and requires 256 GRFs for the forced arm.
The compiler may choose either register count for the default arm.

## Comparisons

Each workload searches the same legal tile, warp, stage and head-group grid
in both arms, with three shuffled timing rounds. Every launchable candidate
is checked against the existing chunked PyTorch attention reference at the
unchanged BF16/FP8 tolerances. Comparisons then freeze their configurations:

1. **Independently tuned winners:** compare each arm's best passing candidate.
   This tests whether adding forced 256 gives tuning a better configuration
   than default-only tuning can find in this grid.
2. **Fixed launch settings:** hold every other parameter fixed and switch only
   the GRF override, at both winning configurations. This tests the effect of
   the override itself. If both winners already share the same settings, the
   first comparison serves both purposes. Unavailable counterparts are reported.

Each comparison uses six alternating timing pairs with the existing cache-cleared
GPU graph timer. A win requires more than a 3% median latency reduction and a
reduction over 3% in at least five of the six pairs. All comparisons are replayed
in a fresh process with the same deterministic inputs and frozen configurations,
reversing which arm goes first. A decisive result must repeat; otherwise it is
labeled `unconfirmed_default` or `unconfirmed_256`. Native binaries must match
across processes. Identical binaries between arms are classified as a tie even
if measured timings differ, so timing noise cannot imply a compiler-option benefit.

## Coverage and artifacts

Resource failures and incorrect candidates are excluded and recorded. A workload
without a passing candidate in either arm is `unavailable`, not a mode win.
Unexpected workload errors are recorded and the next workload is attempted.
Candidate correctness failures make the job fail **after** collecting the rest
of its workload results. FP8 still runs after a BF16 failure if manifest generation
succeeded. No tolerances are relaxed.

Artifacts include per-dtype `summary.json`, `workloads.json` and `comparison.csv`;
per-workload candidate measurements in `search.json`; frozen cases/configurations
in `frozen.json`; initial and confirmed comparison records; native binaries;
and environment, package, source revision and complete manifest records.
Summaries report requested/processed/completed coverage and correctness exclusions
alongside wins and ties. Inspect the detailed fixed-settings comparisons before
attributing a tuned-winner improvement solely to the override.

Results apply to the finite grid, runner compiler/driver, tested inputs and
passing candidates. They do not prove equivalence for untested workloads,
other devices, or the single configuration a shared runtime key might require.

## Local validation and eventual dispatch

```sh
python scripts/vllm/ua_tuning/tune_ci.py --check-patch
python -m pytest -s --tb=short scripts/vllm/ua_tuning/test_compare_grf.py
```

The workflow uses the standard `vLLM benchmarks` entry and run title on this
experiment branch. It is manual and report-only, with read permissions and no
config publication or PR job. After the local branch has been reviewed and publication
and CI dispatch have been explicitly authorized, its dispatch command is:

```sh
gh workflow run vllm-benchmarks.yml \
  --repo intel/intel-xpu-backend-for-triton \
  --ref test/ua-grf-default-vs256-20261110 \
  -f runner_label=b580 -f device=xpu:0 -f part=all
```
