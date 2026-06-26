# RUN — TMA Descriptor Update Benchmark

End-to-end: setup → smoke → calibrate → benchmark. Driven by `run_benchmark.sh`.
Needs a **Hopper (sm_90) or Blackwell (sm_100)** GPU (TMA requires compute
capability ≥ 9.0; won't run on A100/A10G/L4).

The optimization is the `loop-recreated-only` mode of the existing
`triton-rewrite-tensor-descriptor-to-pointer` pass. It demotes only the
`tt.make_tensor_descriptor`s that are recreated inside a loop **and** cannot be
hoisted — i.e. an operand stays loop-varying even after LICM — to pointer
load/stores (the fast path); hoistable and out-of-loop descriptors keep the TMA
path. On Hopper+ the NVIDIA pipeline runs this mode **by default**; pre-Hopper
(and the force knob) run the pass in its original demote-everything mode.

The three benchmark cases:

| case | what | how |
|---|---|---|
| `base` | TMA, optimization **OFF** — raw `tensormap_create` per iteration | `TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1` |
| `tofp` | manual tensor-of-pointers kernel (no descriptor) | `paged_kv_load_pointer_kernel` |
| `opt`  | TMA, optimization **ON** — the pass demotes the loop-recreated descriptor to pointers | default compile |

`base` and `opt` are the **same kernel**; only the compiler pass differs. Key
comparison: **base vs opt** (does the pass help?). `opt` vs `tofp` should be near
parity (opt *is* the pointer lowering, applied by the compiler).

> **Requires the compiler change built in.** The `loop-recreated-only` option on
> `RewriteTensorDescriptorToPointer.cpp` + the `make_ttir` gate must be compiled
> into `libtriton`. If you only did `pip install -e .` without rebuilding after
> the change, rebuild first.

---

## 1. Setup (fresh box)

**Order matters:** torch pins/bundles `triton==3.2.0` (too old — no TMA, no pass).
Install torch FIRST, your Triton build LAST, or torch shadows/downgrades it.

```bash
git clone https://github.com/triton-lang/triton.git   # or your fork
cd triton
python -m venv .venv --prompt triton && source .venv/bin/activate
pip install -r python/requirements.txt                 # build-time deps

# torch FIRST (drags in the 3.2.0 wheel — expected, overridden next)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# your Triton LAST (editable install; rebuilds C++/LLVM incl. the new pass)
pip install -e .        # older layout: `pip install -e python`
```

`pip install -e .` prints a non-fatal `torch requires triton==3.2.0` warning —
ignore it. A later torch (re)install re-triggers the downgrade → re-run
`pip install -e .`.

Verify the right Triton + the pass are present:
```bash
python3 -c "import triton, os; print(os.path.realpath(triton.__file__), triton.__version__)"
#   -> path inside YOUR repo; version NOT 3.2.0
python3 -c "import triton; print(hasattr(triton, 'set_allocator'))"   # -> True
```

### Porting the pass to a clean upstream checkout

The optimization is six files (all outside `tma-update/`). Copy `tma-update/`
into the upstream repo root, then apply the committed snapshot patch via the
wrapper (default target is the parent of `tma-update/`, i.e. the repo root):

```bash
tma-update/run_benchmark.sh --patch --patch-check   # dry-run
tma-update/run_benchmark.sh --patch                 # apply
pip install -e .                                    # REBUILD (C++) — from repo root
```

`--triton-root DIR` targets a different checkout; `--patch-reverse` un-applies.
(These forward to `apply_pass_patch.sh`, which always dry-runs first and refuses
to half-apply.)

`tma-loop-td-pass.patch` covers 7 files: `Passes.td` (the `loop-recreated-only`
option), `RewriteTensorDescriptorToPointer.cpp` (the selective-legality driver +
LICM-invariance check), `python/src/passes.cc` (binding now takes the bool
option), `python/triton/knobs.py` (both knobs), the `make_ttir` gate in
`third_party/nvidia/backend/compiler.py`, the AMD backend (unconditional
demote-all), and a lit test. If upstream drifted and it won't apply, apply the
edits by hand (patch header lists them).

> The C++ pieces (.cpp/.td/CMake/binding) require a `pip install -e .` rebuild;
> the Python pieces (knobs, gate) take effect without one — but the pass they
> call won't exist until the C++ is built.
>
> `gen_knob_patch.py` is the OLD single-knob generator (superseded by this
> patch, which includes the pass and both knobs).

---

## 2. Smoke test — the gate before benchmarking

```bash
tma-update/run_benchmark.sh --smoke-test
```
This is the one check to trust before running the benchmark. It verifies **two**
independent things and only passes if both hold:

1. `[smoke:tma] PASS` — kernel output matches the torch reference (correctness).
2. `[smoke:opt-toggle] PASS` — the optimization **measurably toggles**: default
   compile demotes the descriptor (no `tensormap_create` in the IR); with the
   disable knob the raw `tensormap_create` is preserved. This is a compile-only
   IR check — it catches "pass not built into libtriton" or "disable knob
   broken", the two failures that would silently make base==opt.

If smoke is green, base-vs-opt is a valid A/B — go run the benchmark.

Per-case / options:
```bash
python3 tma-update/paged_kv_load.py smoke --case tma      # correctness + opt-toggle
python3 tma-update/paged_kv_load.py smoke --case pointer  # correctness only (no descriptor)
python3 tma-update/paged_kv_load.py smoke --no-check-opt   # skip the toggle check
```

---

## 3. Calibrate (pick warmup / trials from measured variance)

```bash
tma-update/run_benchmark.sh --calibrate                     # case 'base', burst 500
tma-update/run_benchmark.sh --calibrate --num-kv-blocks 1024 # scaled workload
```
Prints steady-state median/std/**cv%** and a `>>> RECOMMEND: --warmup N --trials M`
line. If cv is high, scale the workload (§4) and/or lock clocks (`nvidia-smi -lgc`).

---

## 4. Scale the workload (lift signal above launch/timer noise)

Decide the workload size **before** benchmarking. At the default the kernel is
small, so it's dominated by launch overhead and timer granularity — the
per-iteration descriptor cost barely registers and variance is high. Scaling up
raises the signal so `base` vs `opt` becomes a clean ratio.

- `--num-kv-blocks N` — loop trip count; scales the per-iteration descriptor cost
  (the thing measured). **The knob to grow** — best fix for a noisy tiny kernel.
  Defaults to 512.
- `--num-seqs N` — grid size / CTAs; fills the GPU, amortizes launch overhead.

No regeneration step is needed for any case — all three JIT-recompile at the
requested size (there is no pre-built PTX to keep in sync anymore).

---

## 5. Benchmark

All three cases + comparison table:
```bash
tma-update/run_benchmark.sh --bench --warmup 100 --trials 500
tma-update/run_benchmark.sh --bench --num-kv-blocks 1024 --warmup 100 --trials 500
```

A single case (no table):
```bash
tma-update/run_benchmark.sh --bench --case base --warmup 100 --trials 500   # opt OFF
tma-update/run_benchmark.sh --bench --case tofp --warmup 100 --trials 500   # manual pointer
tma-update/run_benchmark.sh --bench --case opt  --warmup 100 --trials 500   # opt ON
```
Each line reports `median / min / max / std / cv%`. The table prints:
**base vs opt** (does the optimization help?), base vs tofp, and opt vs tofp
(expected ≈ parity).

---

## Offline (no GPU): compile / inspect IR

```bash
tma-update/run_benchmark.sh --generate --arch sm100   # -> ptx-opt-test/*.ptx (+ ttir/ttgir/llir)
```
To confirm the pass fires: `--generate` at default shows **no** `tensormap_create`
in the ttgir (descriptor demoted); with
`TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1` the `tensormap_create` reappears.

## Wrapper flags

`--generate | --smoke-test | --bench | --calibrate | --patch` (pick one) ·
`--case all|base|tofp|opt` · `--warmup N` · `--trials N` · `--burst N` ·
`--num-kv-blocks N` · `--num-seqs N` · `--arch sm90|sm100` ·
`--triton-root DIR` · `--patch-check` · `--patch-reverse` ·
`--python BIN` (or `TMA_UPDATE_PYTHON=`) · `--out-dir DIR`

## Knobs (NVIDIA make_ttir gate)

- `TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1` — Hopper+: skip the rewrite
  entirely, keep raw per-iteration `tensormap_create` (this is what `--case base`
  sets).
- `TRITON_INTEL_NVIDIA_FORCE_TD_TEST=1` — force the *full* demote-everything
  rewrite (`loop-recreated-only=false`) unconditionally, even on Hopper+ (debug;
  not used by the three cases).

## Gotchas

- **Pre-Hopper (A100/A10G/L4):** the kernel's TMA instructions won't assemble;
  and there `make_tensor_descriptor` is already rewritten to pointers regardless.
  Use a real Hopper/Blackwell GPU.
- **Pass not built:** if `base` and `opt` give identical numbers *and* the ttgir
  still shows `tensormap_create` under default compile, the pass isn't in your
  `libtriton` — rebuild (`pip install -e .`).

See `ptx-opt-test/` for the earlier PTX-level A/B study (the hand/scripted hoist,
now superseded by the compiler pass) and `design-hypothesis.md` for background.
