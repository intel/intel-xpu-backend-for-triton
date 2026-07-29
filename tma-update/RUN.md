# RUN — TMA Descriptor Update Benchmark

End-to-end: setup → smoke → calibrate → benchmark. Driven by `run_benchmark.sh`
for the `paged_kv_load` reproducer (§1–5), and by `run_unified_attention.sh` for
vLLM's real `unified_attention` kernel (§6). Needs a **Hopper (sm_90) or
Blackwell (sm_100)** GPU (TMA requires compute capability ≥ 9.0; won't run on
A100/A10G/L4).

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

## 6. Unified attention — the real vLLM kernel

`run_unified_attention.sh` runs the same three cases against vLLM's
`unified_attention` instead of the `paged_kv_load` reproducer. The kernel is
**imported from the installed vLLM**, never vendored — see *NVIDIA vLLM install*
below. `--probe` is the only mode that works without a GPU or a good install;
run it first.

```bash
tma-update/run_unified_attention.sh --probe        # what does this vLLM support?
tma-update/run_unified_attention.sh --ir           # does the pass apply? selectively?
tma-update/run_unified_attention.sh --smoke-test   # correctness vs torch
tma-update/run_unified_attention.sh --bench --warmup 100 --trials 500
```

`--bench` runs `--ir` first and refuses to time anything if the pass didn't
change the IR.

### `--td-mode` — which descriptors the kernel creates

The kernel gates its descriptors behind two separate constexprs, so it has two
useful configurations:

| mode | kwargs | descriptors | what it proves |
|---|---|---|---|
| `kv` (default) | `use_td=True` | in-loop K/V only | The pass **fires** — every descriptor is loop-recreated, so `tensormap_create` goes to **0**. |
| `all` | `+ use_td_qo=True` | also out-of-loop Q/O | The pass is **selective** — Q/O are hoistable and must survive, so the count **drops but stays > 0**. |

`use_td_qo` defaults to `False` in the kernel and the vLLM wrapper never passes
it. Whether the pin accepts it as a kwarg is reported by `--probe`, not assumed;
if it doesn't, `--td-mode all` is unavailable and only the fires-check applies.

### Reading `--ir`

Prints the `tensormap_create` count for `base` (pass off) and `opt` (pass on).
Two failure modes, distinguished in the output:

- `base == 0` — the TMA path never ran. Either `use_td` didn't reach the kernel
  (a platform gate in the kernel module — `--probe` flags this), or the GPU is
  pre-Hopper, where `make_ttir` demotes everything regardless.
- `opt == base` — the pass changed nothing: not built into `libtriton`, or no
  descriptor here classifies as loop-recreated.

`--ir` forces `TRITON_ALWAYS_COMPILE=1`. Without it a cache hit skips the
pipeline entirely and both cases report a stale count, silently making
`base == opt`.

### Unified-attention wrapper flags

`--probe | --ir | --smoke-test | --bench` (pick one) ·
`--case all|base|tofp|opt` · `--td-mode kv|all` · `--warmup N` · `--trials N` ·
`--no-verify` · `--python BIN` (or `TMA_UPDATE_PYTHON=`)

Unlike `run_benchmark.sh`, preflight **hard-fails** below sm_90 rather than
warning: there all three cases compile to the same code, so the table would be
noise.

> **Scope note.** The `loop-recreated-only` mode is NVIDIA-only. Intel's copy of
> the pass has no such option and `third_party/intel/backend/compiler.py` calls
> it with no argument, demoting every descriptor unconditionally. So this
> benchmark cannot regress XPU — and the XPU benchmark cannot validate the
> change.

## NVIDIA vLLM install

`scripts/vllm/install-vllm.sh` hardcodes `VLLM_TARGET_DEVICE=xpu` and pulls
`vllm-xpu-kernels`; it will not work here. Do it by hand — we only need vLLM's
Python module tree, since the benchmark just imports one `@triton.jit` kernel.

**Same ordering rule as §1:** torch first, your Triton **last**. Every pip step
below can silently drag in the `triton==3.2.0` wheel.

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout "$(cat /path/to/intel-xpu-backend-for-triton/scripts/vllm/vllm-pin.txt)"

# strip anything that would reinstall torch/triton over your build
sed -i -E '/^(torch|triton|xformers)/d; /^--extra-index-url.*download\.pytorch\.org/d' \
    requirements/common.txt requirements/cuda.txt
pip install -r requirements/cuda.txt

# 'empty' skips building vLLM's C++/CUDA kernels — we import a Triton kernel, not
# vLLM's ops. --no-deps stops vLLM's own torch/triton pins from being honoured.
VLLM_TARGET_DEVICE=empty pip install --no-deps --no-build-isolation -e .
```

If your pin rejects `VLLM_TARGET_DEVICE=empty`, use `VLLM_USE_PRECOMPILED=1`
instead of building the CUDA kernels from source. `--no-deps` means a missing
transitive import surfaces at `import vllm` — install those one at a time with
`pip install --no-deps <pkg>`.

**Do not run `scripts/vllm/vllm_xpu_patch.py`** — it rewrites CUDA→XPU in vLLM's
test and layer directories. (It never touches `vllm/v1/attention/ops/` anyway,
which is exactly why this kernel is portable.)

`scripts/vllm/vllm-fix.patch` is optional. Three of its four hunks are XPU/test
fixes that are harmless but unnecessary on CUDA; the fourth is a device-agnostic
Triton-JIT annotation fix to `triton_unified_attention.py` (`tl.int64 = None` →
`int | None = None`, plus the matching wrapper change). Apply it if `--smoke-test`
dies with a `TypeError` about `NoneType` and an integer:

```bash
git apply /path/to/intel-xpu-backend-for-triton/scripts/vllm/vllm-fix.patch
```

Verify — the second line is the one people skip and regret:

```bash
tma-update/run_unified_attention.sh --probe    # imports the kernel, reports use_td/use_td_qo
python3 -c "import triton, os; print(os.path.realpath(triton.__file__), triton.__version__)"
```

If that triton path is no longer inside your repo, re-run `pip install -e .`
there.

---

## Offline (no GPU): compile / inspect IR

```bash
tma-update/run_benchmark.sh --generate --arch sm100   # -> ptx-opt-test/*.ptx (+ ttir/ttgir/llir)
```
To confirm the pass fires: `--generate` at default shows **no** `tensormap_create`
in the ttgir (descriptor demoted); with
`TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1` the `tensormap_create` reappears.

## Wrapper flags (`run_benchmark.sh`)

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
