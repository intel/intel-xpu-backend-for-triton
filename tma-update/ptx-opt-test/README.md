# Smoke Test & Benchmark — How to Run

Benchmark four ways of expressing the paged-KV load, to answer "is TMA worth it,
and does our optimization help":

| case | name | what | mechanism |
|---|---|---|---|
| #1 | `base` | baseline TMA — `tensormap_create` per iteration | default JIT |
| #2 | `tofp` | tensor-of-pointers — explicit `tl.load`, no descriptor | `paged_kv_load_pointer_kernel` |
| #3 | `fallback` | TD source, but rewritten to pointers (the non-TMA fallback) | `TRITON_INTEL_NVIDIA_FORCE_TD_TEST=1` |
| #4 | `opt` | our optimization — hoisted build + per-iter `global_address` replace | inject `optimized.ptx` via override |

Key comparisons: #1vs#4 (does the opt help the TMA path), #1vs#2 (is TMA worth
it vs pointers), #2vs#3 (is the fallback as good as hand-written pointers),
#4vs#2 (does optimized TMA finally beat pointers).

> **You need a Hopper (sm_90) or Blackwell (sm_100) GPU.** TMA
> (`tensormap.*`, `cp.async.bulk.tensor`) requires compute capability ≥ 9.0.
> It will **not** assemble on Ampere (A100/A10G, sm_80/86) or Ada (L4/L40S,
> sm_89). Cheapest valid smoke-test target = **H100**. The kernel body is
> byte-identical between sm_90 and sm_100 (only the `.version`/`.target` header
> differs), so Hopper is a faithful smoke test of the mechanism.

## Prerequisites (fresh box)

The benchmark needs **two** things in the venv: a recent Triton with device-side
tensor-descriptor support (`triton.set_allocator` + TMA), and **torch** (for the
tensors + correctness reference + CUDA-event timing).

**Order matters.** `torch` pins `triton==3.2.0` and bundles that wheel — which
predates the TMA-descriptor feature. So install torch FIRST and your Triton
build LAST, or torch's resolver will shadow/downgrade your build.

```bash
# --- build Triton from source (upstream layout) ---
git clone https://github.com/triton-lang/triton.git   # or your fork
cd triton
python -m venv .venv --prompt triton
source .venv/bin/activate
pip install -r python/requirements.txt                 # build-time deps

# --- torch FIRST (drags in a triton 3.2.0 wheel — expected, overridden next) ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- your Triton LAST (editable install supersedes the 3.2.0 wheel) ---
pip install -e .        # newer Triton: pyproject at repo root; older: `pip install -e python`
```

`pip install -e .` prints a dependency-conflict WARNING
(`torch ... requires triton==3.2.0, but you have 3.8.0`). This is **non-fatal** —
the editable install still wins; torch imports and uses your build fine. Any
later `pip install`/upgrade of torch can re-trigger the 3.2.0 downgrade → just
re-run `pip install -e .`.

Verify the right Triton is imported (not the wheel):

```bash
python3 -c "import triton, os; print(os.path.realpath(triton.__file__), triton.__version__)"
#   -> path must be inside your repo; version 3.8.0+... (NOT site-packages / 3.2.0)
python3 -c "import triton; print(hasattr(triton, 'set_allocator'))"   # -> True
python3 -c "import torch; print(torch.cuda.get_device_capability())"  # -> (9,0) or (10,0)
```

`run_benchmark.sh` runs a **preflight** that checks all of the above (except for
`--generate`, which is offline) and fails with a fix hint if the env is wrong —
so you don't have to remember this list, but this is what it's checking.

## TL;DR — the wrapper does everything

`../run_benchmark.sh` orchestrates the whole flow (override key discovery,
injection, correctness gate, timing, cleanup):

```bash
# Offline (no GPU): compile the kernel to baseline PTX.
tma-update/run_benchmark.sh --generate --arch sm100

# On the GPU box: does it run + is it correct?
tma-update/run_benchmark.sh --smoke-test

# On the GPU box: all four cases + comparison table.
tma-update/run_benchmark.sh --bench

# On the GPU box: a SINGLE case (no table).
tma-update/run_benchmark.sh --bench --case base
tma-update/run_benchmark.sh --bench --case tofp
tma-update/run_benchmark.sh --bench --case fallback
tma-update/run_benchmark.sh --bench --case opt
```

Useful flags: `--arch sm90|sm100`, `--case all|base|tofp|fallback|opt`,
`--optimized <file>`, `--python <bin>` (or `TMA_UPDATE_PYTHON=`),
`--warmup N`, `--trials N`, `--keep-override`.

Notes:
- The wrapper asserts the `"Overriding kernel"` line fired for the `opt` case;
  if it didn't, it warns that #4 is silently the baseline (`src.hash()` mismatch).
- Only the `opt` case needs the override-key discovery + the `optimized.ptx`
  file; `base`/`tofp`/`fallback` skip both.
- `fallback` (#3) sets `TRITON_INTEL_NVIDIA_FORCE_TD_TEST=1`, a debug knob in
  `compilation_knobs` that widens the NVIDIA `make_ttir` gate so the
  `make_tensor_descriptor` → pointer rewrite runs on TMA-capable targets.

The sections below document the **manual** equivalent of what the wrapper does,
for when you need to debug a step.

## Files

- `paged_kv_load_kernel.ptx` — baseline (sm_100, generated offline on XE4).
- `paged_kv_load_kernel.optimized.ptx` — optimized PTX, regenerable via `ptx_hoist.py`.
- `ptx_hoist.py` — scripts the baseline → optimized hoist transform.
- `../paged_kv_load.py` — the kernel(s) + `run`/`smoke_test`/`bench` (Python `--case tma|pointer|fallback`).
- See `ANNOTATED.md` / `NOTES.md` for what the edit does and why.

## 0. Pick the PTX for your GPU

The committed `.ptx` files target **sm_100** (`.version 9.3`, `.target
sm_100a`). On an **H100 (sm_90)** you must regenerate/retarget — the simplest,
least error-prone path is to regenerate the baseline *on the GPU box* so the
`.version` matches its local ptxas, then re-apply the same edit:

```bash
# On the GPU box, in the repo root:
TMA_UPDATE_OUT_DIR=/tmp/tma TMA_UPDATE_ARCH=sm90 \
TMA_UPDATE_COMPILE_ONLY=1 TMA_UPDATE_STOP_AFTER=ptx \
  python tma-update/paged_kv_load.py
# -> /tmp/tma/paged_kv_load_kernel.ptx  (baseline, sm_90)
```

To produce the sm_90 **optimized** variant, apply the same two changes the
sm_100 optimized file documents (see its header + `ANNOTATED.md`): move the 12
invariant `tensormap.replace.tile.*` + the zero-fill above `$L__BB0_1`, relocate
the descriptor staging to a non-overlapping SMEM offset, and leave only
`global_address` replace + `cp_fenceproxy` + `acquire` in the loop. (Diffing the
committed sm_100 pair shows exactly which lines move.)

## 1. Find the override directory (one-time, per kernel)

The override dir is keyed by `base32(src.hash())`. Easiest way to discover it:
dump IR once — the dump dir uses the **same** key as the override dir.

```bash
# On the GPU box:
TRITON_KERNEL_DUMP=1 python tma-update/paged_kv_load.py
ls ~/.triton/dump/            # -> one dir, e.g. ABCD...XYZ  (the base32 key)
KEY=$(ls -t ~/.triton/dump/ | head -1)
echo "override key = $KEY"
```

## 2. Inject a PTX variant

Place your chosen `.ptx` at `~/.triton/override/$KEY/paged_kv_load_kernel.ptx`
(filename must be `<kernel_name>.ptx`), then run with override enabled:

```bash
mkdir -p ~/.triton/override/$KEY
cp /tmp/tma/paged_kv_load_kernel.ptx \
   ~/.triton/override/$KEY/paged_kv_load_kernel.ptx

TRITON_KERNEL_OVERRIDE=1 TRITON_ALWAYS_COMPILE=1 \
  python tma-update/paged_kv_load.py
```

**Confirm the override actually fired** — Triton prints:

```
Overriding kernel with file .../paged_kv_load_kernel.ptx
```

If you don't see that line, the override missed (wrong KEY, wrong filename, or
source changed → new `src.hash()`); you'd be silently benchmarking the baseline.
This is the #1 footgun.

## 3. Smoke test (does it run / is it correct)

`paged_kv_load.py smoke [--case tma|pointer|fallback]` launches once and
verifies the output against a torch reference (`gather kv_cache[block_tables]`)
with `torch.testing.assert_close`. It prints `[smoke:<case>] PASS/FAIL`.

```bash
python tma-update/paged_kv_load.py smoke --case tma       # baseline TMA
python tma-update/paged_kv_load.py smoke --case pointer   # tensor-of-pointers
TRITON_INTEL_NVIDIA_FORCE_TD_TEST=1 \
  python tma-update/paged_kv_load.py smoke --case fallback # TD->pointer fallback
```

For the **optimized** PTX, smoke-test it by injecting via override (§1-2) and
running any `smoke`/`bench`. If ptxas rejects the generated PTX, the assembly
error surfaces here — the first real ptxas validation (XE4 has no ptxas).

## 4. Benchmark (timing)

`paged_kv_load.py bench` does CUDA-event timing **behind a correctness gate**
(it verifies vs the torch reference first, then warms up, then times):

```bash
python tma-update/paged_kv_load.py bench --case tma --warmup 50 --trials 100
# -> [bench:tma] correctness gate PASS
#    [bench:tma] median 0.0XXXX ms  min ...  max ...
```

But prefer `../run_benchmark.sh --bench` (or `--bench --case <name>`), which runs
all four cases, injects the optimized PTX for `opt`, gates each on correctness,
and prints the comparison table. The manual single-case form above is for
debugging one case in isolation.

### The measurement that decomposes the tradeoff

Sweep `NUM_KV_BLOCKS` in `paged_kv_load.py` and plot time vs. trips for #1 and
#4: **slope** = per-iteration descriptor cost (expect baseline slope > opt),
**intercept gap** = the one-time hoisted build cost.

### The measurement that actually decomposes the tradeoff

Sweep the loop trip count (`NUM_KV_BLOCKS` in `paged_kv_load.py`) and plot time
vs. trips for both variants:

- **slope** = per-iteration descriptor cost → expect baseline slope > optimized.
- **intercept gap** = one-time hoisted-build cost of the optimized variant.

That separates "saved per-iter work" from "fixed preheader cost" — the core
question of whether the optimization pays off.

## 5. Confirm ptxas didn't undo the hoist

ptxas is an optimizing assembler and could re-sink the hoisted instructions.
Inspect the SASS:

```bash
# find the cached cubin after an override run, or compile the ptx directly:
ptxas -arch=sm_90 paged_kv_load_kernel.optimized.ptx -o /tmp/k.cubin
cuobjdump -sass /tmp/k.cubin | grep -nE "UTMA|tensormap|BAR|fence" | head
```

If the descriptor-build instructions reappear inside the loop body in SASS, the
hoist didn't survive — which is itself a finding (argues the optimization must
live as a real op + scheduling barrier, not a PTX rewrite).

## Caveats

- Single-buffer microbench: shows per-iter descriptor cost cleanly, but not the
  occupancy interaction (the +128 B persistent SMEM). That needs the pipelined
  unified-attention kernel — phase 2.
- Hopper gives correctness + per-iter signal; confirm absolute numbers and the
  SMEM/occupancy verdict on the actual Blackwell target.
