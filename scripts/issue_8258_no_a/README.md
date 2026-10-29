# Focused baseline versus A-prefetch removal

The initial authorized run, [40775539032](https://github.com/intel/intel-xpu-backend-for-triton/actions/runs/40775539032), failed after autotuning the first case: the parser supported GenISA prefetch intrinsics, but the CI compiler emitted the SPIR-V subgroup 2D prefetch builtin. No policy-comparison measurements completed. The benchmark-wheel warning was handled by a successful fallback and did not cause the failure.

The first repair supports both prefetch interfaces with the same strict ownership checks. It also saves original LLVM IR before parsing and prints the final worker log lines on failure. The authorized replacement run, [40883989884](https://github.com/intel/intel-xpu-backend-for-triton/actions/runs/40883989884), completed one case and failed on the second case's unmodified `stock_driver` correctness check. At M512/N4096/K8192, one of 2,097,152 outputs was 0.0037703514 versus PyTorch's 0.0036621094, outside atol1e-4/rtol1e-3. The saved evidence does not establish the FP64 value, so rounding is a hypothesis rather than an established explanation.

The second local repair retains those tolerances and strict case rejection. It records up to 16 FP64 dot-product diagnostics plus the corresponding CPU input vectors for isolated full-output mismatches. Diagnostics never override a failed check. Each failed/timed-out worker is recorded and the remaining manifest cases are still attempted; failures are not retried. The report excludes failed cases, lists them and their population counts explicitly, and exits nonzero if any case failed. A completed attempt of the scope is distinct from an all-valid study. Another push and workflow dispatch require explicit approval; no third run or local GPU benchmark has been performed.

- Source baseline: `06610e5d96e1d7fb75aa7aac61e4ef61af54a4d3`.
- Intended isolated branch: `ci/issue-8258-no-a-prefetch-20261028`.
- Workflow path: `.github/workflows/triton-benchmarks.yml`, replaced on the isolated branch with `workflow_dispatch` only.
- Exactly one B580 job, 180-minute cap, no matrix fanout or unrelated benchmark tests.
- 54 logical shapes and 30 extra stride/address cases: 84 manifest cases total.
- Only two policies: original prefetch and remove A prefetch. All B-prefetch calls remain byte-identical in LLVM IR.
- Evaluate the freshly autotuned stock configuration per actual layout and fixed256×256×32/two-stage/32warps/256GRF; deduplicate identical configurations. At most336 configuration/policy cases before no-op/configuration deduplication.
- An extra unmodified default driver-SPIR-V reference lane per case exposes differences between that path and the explicit native compilation used for exact-code auditing. At most420 measured lanes before deduplication. This is a calibration control, not another candidate policy.

`transform.py` traces each prefetch base back to the first A/B user pointers, permits compiler-added implicit scratch pointers, and fails on ambiguous/unsupported ownership. It removes only A-owned void prefetch call lines. Missing A prefetch is reported as a no-op. The unchanged LLVM→SPIR-V roundtrip must reproduce the original bytes.

The supported forms are the GenISA intrinsic (integer base address in argument zero) and the exact mangled `__spirv_Subgroup2DBlockPrefetchINTEL` builtin (pointer base in argument four). CPU regression cases cover both forms, mixed forms, unknown/C/scratch bases, ambiguous ownership and unsupported builtin signatures.

Both matched native policies use the backend's `make_zebin` path with the same source metadata/options. Saved native files are the exact bytes loaded by the explicit native runners, and include disassembly, prefetch counts, and loaded resource metrics. A default driver-SPIR-V reference is measured separately. Report stock-relative gains, matched-native gains, and native-baseline/driver differences distinctly; a compilation-path difference alone is not evidence for A-removal benefit.

Each manifest case runs in a separate process. That avoids reusing an M/N/K-only autotune choice across different strides, releases resources between cases, and preserves the same allocations across policies within a case. The stock search retains its16 candidates. All candidates receive full-output correctness checks using the stock contract (atol1e-4/rtol1e-3), CPU-FP64 sampled corroboration, finite-output checks and final read-only-input checks.

Two short randomized selection rounds are separate from three randomized measurement rounds. The selection pool contains the stock reference and active A-removal alternatives; it does not choose the best policy after seeing measurement results. GEMM timing uses raw XPU profiler events, seven launches per measurement group, preallocated output, and an excluded256MiB flush. Correctness is checked before and after measurement. No current-tile diagnostic, B-removal, both-removal, or cache-bypass policy is included.

`analyze.py` requires records accounting for all84 requested cases, including explicit failure records for rejected cases. Performance aggregates cover eligible complete cases only and record every apparent regression, no-op, selected policy, stock-relative and matched-native effect and calibration difference. Invalid counts remain visible for each population; any invalid case makes the workflow fail after writing the report. Its break-even figure uses measured study preparation/selection time including validation and evidence overhead; it is not an estimate of production tuner cost. Contiguous and layout-stress populations are separate. Three rounds screen candidates; fresh-allocation confirmation would be a separate, permission-gated follow-up.

Local validation:

- CPU ownership/no-op/unsupported-provenance checks and all76 saved LLVM fixtures passed.
- Synthetic reporting checks ensure a policy selected earlier is still reported as a regression when later measurement loses; no-op aliases are retained.
- Four stock-kernel ASTs compiled with an explicit saved B580 target and hardware queries disabled: one-stage/no-op, two-stage original, N6080 tail and padded strides. Unchanged SPIR-V roundtrips matched exactly; only A prefetch disappeared. Seven offline native images passed `ocloc validate`.
- These offline checks used the available local Triton3.8 compiler. They are not a build of the pinned CI compiler and are not GPU correctness/performance results. The workflow builds the pinned source and runs the GPU checks only after approval.

The initial offline compile attempt exposed a local XPU-availability query despite the explicit target. The offline checker now fixes the architecture, hides devices and disables Torch availability queries. It does not launch kernels. Runtime execution of the new harness has not been tested locally, honoring the restriction on local GPU benchmarks.

Repair validation used the existing local Triton 3.9 build at source commit `d08bc5046315e7f3fef985eaae4c63ee1d48a6dd`, with devices hidden and availability queries disabled. Five stock AST cases include the exact first failing shape at two stages, a one-stage no-op, the original large shape, an N6080 tail and padded strides. Four emit the SPIR-V builtin: A2/B2 becomes A0/B2, with unchanged baseline SPIR-V roundtrips. All nine native images pass `ocloc validate` and disassembly; B-prefetch instructions remain in every treatment. Four saved older-format LLVM modules also pass. This is CPU compilation evidence with an existing build, not a rebuild of the exact CI commit or GPU correctness/performance evidence.

Remote approval must cover pushing this prepared isolated branch and dispatching exactly one instance of the scoped workflow. No automatic rerun is included.

Second-repair CPU checks cover cancellation diagnostics with a known FP64 result, rejection despite a favorable FP64 diagnostic, exclusion of failed cases from aggregates, a population with no eligible cases, and continuing exactly the84 manifest cases after simulated worker failure and timeout. No GPU libraries are imported by these checks. The changed Python validation/runner path has not been exercised on a GPU locally.
