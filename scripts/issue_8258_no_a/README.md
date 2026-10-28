# Focused baseline versus A-prefetch removal

Prepared locally for approval. No branch push, workflow dispatch, remote issue/PR, or new local GPU benchmark has been performed for this study.

- Source baseline: `06610e5d96e1d7fb75aa7aac61e4ef61af54a4d3`.
- Intended isolated branch: `ci/issue-8258-no-a-prefetch-20261028`.
- Workflow path: `.github/workflows/triton-benchmarks.yml`, replaced on the isolated branch with `workflow_dispatch` only.
- Exactly one B580 job, 180-minute cap, no matrix fanout or unrelated benchmark tests.
- 54 logical shapes and 30 extra stride/address cases: 84 manifest cases total.
- Only two policies: original prefetch and remove A prefetch. All B-prefetch calls remain byte-identical in LLVM IR.
- Evaluate the freshly autotuned stock configuration per actual layout and fixed256×256×32/two-stage/32warps/256GRF; deduplicate identical configurations. At most336 configuration/policy cases before no-op/configuration deduplication.
- An extra unmodified default driver-SPIR-V reference lane per case exposes differences between that path and the explicit native compilation used for exact-code auditing. At most420 measured lanes before deduplication. This is a calibration control, not another candidate policy.

`transform.py` traces each prefetch base back to the first A/B user pointers, permits compiler-added implicit scratch pointers, and fails on ambiguous/unsupported ownership. It removes only A-owned void prefetch call lines. Missing A prefetch is reported as a no-op. The unchanged LLVM→SPIR-V roundtrip must reproduce the original bytes.

Both matched native policies use the backend's `make_zebin` path with the same source metadata/options. Saved native files are the exact bytes loaded by the explicit native runners, and include disassembly, prefetch counts, and loaded resource metrics. A default driver-SPIR-V reference is measured separately. Report stock-relative gains, matched-native gains, and native-baseline/driver differences distinctly; a compilation-path difference alone is not evidence for A-removal benefit.

Each manifest case runs in a separate process. That avoids reusing an M/N/K-only autotune choice across different strides, releases resources between cases, and preserves the same allocations across policies within a case. The stock search retains its16 candidates. All candidates receive full-output correctness checks using the stock contract (atol1e-4/rtol1e-3), CPU-FP64 sampled corroboration, finite-output checks and final read-only-input checks.

Two short randomized selection rounds are separate from three randomized measurement rounds. The selection pool contains the stock reference and active A-removal alternatives; it does not choose the best policy after seeing measurement results. GEMM timing uses raw XPU profiler events, seven launches per measurement group, preallocated output, and an excluded256MiB flush. Correctness is checked before and after measurement. No current-tile diagnostic, B-removal, both-removal, or cache-bypass policy is included.

`analyze.py` requires exactly84 complete cases and records every apparent regression, no-op, selected policy, stock-relative and matched-native effect and calibration difference. Its break-even figure uses measured study preparation/selection time including validation and evidence overhead; it is not an estimate of production tuner cost. Contiguous and layout-stress populations are separate. Three rounds screen candidates; fresh-allocation confirmation would be a separate, permission-gated follow-up.

Local validation:

- CPU ownership/no-op/unsupported-provenance checks and all76 saved LLVM fixtures passed.
- Synthetic reporting checks ensure a policy selected earlier is still reported as a regression when later measurement loses; no-op aliases are retained.
- Four stock-kernel ASTs compiled with an explicit saved B580 target and hardware queries disabled: one-stage/no-op, two-stage original, N6080 tail and padded strides. Unchanged SPIR-V roundtrips matched exactly; only A prefetch disappeared. Seven offline native images passed `ocloc validate`.
- These offline checks used the available local Triton3.8 compiler. They are not a build of the pinned CI compiler and are not GPU correctness/performance results. The workflow builds the pinned source and runs the GPU checks only after approval.

The initial offline compile attempt exposed a local XPU-availability query despite the explicit target. The offline checker now fixes the architecture, hides devices and disables Torch availability queries. It does not launch kernels. Runtime execution of the new harness has not been tested locally, honoring the restriction on local GPU benchmarks.

Remote approval must cover pushing this prepared isolated branch and dispatching exactly one instance of the scoped workflow. No automatic rerun is included.
