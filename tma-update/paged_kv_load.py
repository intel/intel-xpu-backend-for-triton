"""
Minimal reproducer for the per-iteration `tensormap_create` cost on Hopper+.

The kernel mirrors the inner KV loop from the vLLM unified_attention patch
(benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch).
The descriptor's `base` depends on a value loaded INSIDE the loop, so the
compiler can't hoist the descriptor build. Every other field (shape, strides,
block_shape) is loop-invariant — the only thing that should be re-written each
iteration is `global_address`.

NOT-HOISTABLE BY CONSTRUCTION. The descriptor's `base` is:

    physical_block_idx = tl.load(block_tables_ptr + bt_off + j)   # IV-keyed load
    kv_base = kv_cache_ptr + physical_block_idx * stride_kv_0 + ...

`physical_block_idx` is a runtime tensor read whose value the compiler does
not know, and it changes every iteration. `triton_licm` cannot move a load
that depends on the IV, and the paged-KV access is genuinely non-contiguous
(block_tables[j] can map to any physical block), so it can't be folded into
offsets on a single hoisted descriptor either. The descriptor build MUST live
inside the loop. This file is the reproducer for the in-loop create case —
not the hoistable case.

This module exposes:

  * `paged_kv_load_kernel`           — the @triton.jit kernel.
  * `compile_for_target(arch_str)`   — invoke `triton.compile` against a chosen
                                       GPUTarget. No GPU needs to be present.
                                       Use this from non-Nvidia hosts to dump
                                       Hopper / Blackwell IR.
  * `run()`                          — JIT-launch the kernel on the local GPU
                                       (cuda or xpu). Only works on a real
                                       device.

Run via tma-update/run_blackwell.sh, which wires up TRITON_CACHE_DIR /
TRITON_KERNEL_DUMP / MLIR_ENABLE_DUMP for you.
"""

import os

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget


def _set_allocator(fn):
    """
    Register the global-scratch allocator (needed for on-device TMA descriptors).

    `triton.set_allocator` exists on any Triton new enough to support the
    device-side tensor-descriptor feature this benchmark targets. If it's
    missing, the install is too old — fail with that message rather than
    limping on.
    """
    setter = getattr(triton, "set_allocator", None)
    if setter is None:
        raise RuntimeError(f"triton.set_allocator not found (triton "
                           f"{getattr(triton, '__version__', '?')} at {triton.__file__}). "
                           "This benchmark needs a Triton with device-side tensor-descriptor "
                           "support (make_tensor_descriptor + TMA); the installed version is "
                           "too old. Install the source repo you intend to test "
                           "(e.g. `pip install -e python` from the triton checkout).")
    setter(fn)


@triton.jit
def paged_kv_load_kernel(
    out_ptr,  # [num_seqs, num_kv_blocks, TILE_SIZE, HEAD_SIZE]
    kv_cache_ptr,  # [num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]
    block_tables_ptr,  # [num_seqs, max_blocks_per_seq] int32
    stride_kv_0,
    stride_kv_1,
    stride_kv_2,
    stride_bt_0,
    stride_out_0,
    stride_out_1,
    # Note: the inner-most stride must be a compile-time `1` for
    # tl.make_tensor_descriptor (semantic.py asserts `strides[-1] == 1`).
    # In the vLLM JIT path this falls out of stride specialization; in our
    # AOT compile path we declare it as a constexpr.
    KV_HEAD_IDX: tl.constexpr,
    NUM_KV_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    seq_idx = tl.program_id(0)
    bt_off = seq_idx * stride_bt_0

    for j in range(NUM_KV_BLOCKS):
        # Per-iteration indirection through the block table.
        # `physical_block_idx` is unknown until run time, so `kv_base` is
        # genuinely loop-varying and the descriptor build cannot be hoisted.
        physical_block_idx = tl.load(block_tables_ptr + bt_off + j).to(tl.int64)

        kv_base = (kv_cache_ptr + physical_block_idx * stride_kv_0 + KV_HEAD_IDX * stride_kv_2)

        # Everything below `base=` is loop-invariant. Only `base` changes.
        # Inner-most stride is a literal 1 (HEAD_SIZE-contiguous KV layout).
        kv_desc = tl.make_tensor_descriptor(
            base=kv_base,
            shape=(BLOCK_SIZE, HEAD_SIZE),
            strides=(stride_kv_1, 1),
            block_shape=(TILE_SIZE, HEAD_SIZE),
        )

        tile = kv_desc.load([0, 0])

        out_off = seq_idx * stride_out_0 + j * stride_out_1
        offs_t = tl.arange(0, TILE_SIZE)
        offs_d = tl.arange(0, HEAD_SIZE)
        tl.store(out_ptr + out_off + offs_t[:, None] * HEAD_SIZE + offs_d[None, :], tile)


@triton.jit
def paged_kv_load_pointer_kernel(
    out_ptr,  # [num_seqs, num_kv_blocks, TILE_SIZE, HEAD_SIZE]
    kv_cache_ptr,  # [num_blocks, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]
    block_tables_ptr,  # [num_seqs, max_blocks_per_seq] int32
    stride_kv_0,
    stride_kv_1,
    stride_kv_2,
    stride_bt_0,
    stride_out_0,
    stride_out_1,
    KV_HEAD_IDX: tl.constexpr,
    NUM_KV_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
):
    """
    Case #2: tensor-of-pointers. Identical gather to paged_kv_load_kernel, but
    expressed with explicit pointer arithmetic + tl.load instead of a tensor
    descriptor. No descriptor is ever created — this is what a kernel author
    would write by hand before the TD API, and the reference point for "is TMA
    worth it at all" / "is the TD->pointer fallback as good as hand pointers".
    """
    seq_idx = tl.program_id(0)
    bt_off = seq_idx * stride_bt_0

    offs_t = tl.arange(0, TILE_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE)
    for j in range(NUM_KV_BLOCKS):
        physical_block_idx = tl.load(block_tables_ptr + bt_off + j).to(tl.int64)

        kv_base = (kv_cache_ptr + physical_block_idx * stride_kv_0 + KV_HEAD_IDX * stride_kv_2)
        # tile[t, d] = kv_cache[phys, t, KV_HEAD_IDX, d]
        kv_ptrs = kv_base + offs_t[:, None] * stride_kv_1 + offs_d[None, :]
        tile = tl.load(kv_ptrs)

        out_off = seq_idx * stride_out_0 + j * stride_out_1
        tl.store(out_ptr + out_off + offs_t[:, None] * HEAD_SIZE + offs_d[None, :], tile)


# Default constexpr / signature shape for headless compiles. These match the
# `run()` configuration below so the IR is identical between the two paths.
DEFAULT_CONSTEXPRS = dict(
    KV_HEAD_IDX=0,
    NUM_KV_BLOCKS=512,  # loop trip count — sized to keep the per-iteration
    # descriptor cost well above launch/timer overhead
    BLOCK_SIZE=16,
    TILE_SIZE=16,
    HEAD_SIZE=128,
)

DEFAULT_SIGNATURE = {
    "out_ptr": "*bf16",
    "kv_cache_ptr": "*bf16",
    "block_tables_ptr": "*i32",
    "stride_kv_0": "i64",
    "stride_kv_1": "i64",
    "stride_kv_2": "i64",
    "stride_bt_0": "i64",
    "stride_out_0": "i64",
    "stride_out_1": "i64",
    "KV_HEAD_IDX": "constexpr",
    "NUM_KV_BLOCKS": "constexpr",
    "BLOCK_SIZE": "constexpr",
    "TILE_SIZE": "constexpr",
    "HEAD_SIZE": "constexpr",
}


def compile_for_target(arch="sm100", *, num_warps=4, num_stages=3, stop_after="ttgir", out_dir=None,
                       num_kv_blocks=None):
    """
    Run the Nvidia compile pipeline against an arbitrary GPUTarget — no GPU
    required on the host. We drive the stages manually instead of calling
    `triton.compile`, so we can stop after TTGIR (which is where the
    `ttng.tensormap_create` IR lives). Going below TTGIR on this build hits
    Nvidia C++ passes that aren't compiled into the Intel-fork's libtriton.

    Args:
      arch:          "sm90", "sm100", ...
      num_warps:     backend option
      num_stages:    backend option
      stop_after:    "ttir" or "ttgir" — stage to halt after.
      out_dir:       if given, write the IR after each stage as
                     paged_kv_load_kernel.<stage> into out_dir.
      num_kv_blocks: override the loop trip count baked into the PTX. MUST match
                     the runtime --num-kv-blocks used for the 'opt' override, or
                     the injected PTX won't match the kernel.

    Returns the final IR module string.
    """
    from triton._C.libtriton import ir
    from triton.compiler.compiler import ASTSource, make_backend

    capability = int(arch.removeprefix("sm"))
    target = GPUTarget(backend="cuda", arch=capability, warp_size=32)

    constexprs = dict(DEFAULT_CONSTEXPRS)
    if num_kv_blocks is not None:
        constexprs["NUM_KV_BLOCKS"] = int(num_kv_blocks)

    src = ASTSource(
        fn=paged_kv_load_kernel,
        signature=DEFAULT_SIGNATURE,
        constexprs=constexprs,
    )
    backend = make_backend(target)
    options = backend.parse_options(dict(num_warps=num_warps, num_stages=num_stages))

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()

    module = src.make_ir(target, options, codegen_fns, module_map, context)

    stages = {}
    backend.add_stages(stages, options, src.language)
    metadata = {"target": target, **options.__dict__}

    written = []
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{src.name}.source")
        with open(path, "w") as f:
            f.write(str(module))
        written.append(path)

    last_str = None
    for ext, compile_ir in stages.items():
        module = compile_ir(module, metadata)
        last_str = str(module) if not isinstance(module, (bytes, bytearray)) else None
        if out_dir is not None and last_str is not None:
            path = os.path.join(out_dir, f"{src.name}.{ext}")
            with open(path, "w") as f:
                f.write(last_str)
            written.append(path)
        if ext == stop_after:
            break

    if out_dir is not None:
        print(f"[paged_kv_load] wrote {len(written)} files to {out_dir}:", flush=True)
        for p in written:
            print(f"  {p}", flush=True)

    return last_str


# --- runtime config for the on-GPU paths (smoke test + benchmark) ---
NUM_BLOCKS = 64
KV_HEADS = 4
NUM_SEQS = 8


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "xpu"


def _make_state(seed=0, case="tma", num_seqs=None, num_kv_blocks=None):
    """
    Allocate kernel inputs/outputs once. Returns a dict reused across launches.

    Workload size can be scaled to lift the timing signal above launch/timer
    overhead:
      * num_kv_blocks -> loop trip count (the per-iteration descriptor cost this
        benchmark measures scales directly with this). This is the knob to grow.
      * num_seqs      -> grid size / CTAs (fills the GPU, amortizes launch cost).
    NUM_BLOCKS (physical KV pool) auto-grows so block_tables indices stay valid.
    """
    import torch

    assert case in CASES, f"case must be one of {CASES}, got {case!r}"
    torch.manual_seed(seed)
    # per-state copy so overriding NUM_KV_BLOCKS doesn't mutate the module default
    cfg = dict(DEFAULT_CONSTEXPRS)
    if num_kv_blocks is not None:
        cfg["NUM_KV_BLOCKS"] = int(num_kv_blocks)
    n_seqs = int(num_seqs) if num_seqs is not None else NUM_SEQS
    device = _device()

    # physical block pool must cover the largest index a block_table can hold
    num_blocks = max(NUM_BLOCKS, cfg["NUM_KV_BLOCKS"])

    kv_cache = torch.randn(
        num_blocks,
        cfg["BLOCK_SIZE"],
        KV_HEADS,
        cfg["HEAD_SIZE"],
        dtype=torch.bfloat16,
        device=device,
    )
    block_tables = torch.randint(
        0,
        num_blocks,
        (n_seqs, cfg["NUM_KV_BLOCKS"]),
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty(
        n_seqs,
        cfg["NUM_KV_BLOCKS"],
        cfg["TILE_SIZE"],
        cfg["HEAD_SIZE"],
        dtype=torch.bfloat16,
        device=device,
    )

    _set_allocator(lambda size, alignment, stream: torch.empty(size, device=device, dtype=torch.int8))

    return dict(cfg=cfg, device=device, kv_cache=kv_cache, block_tables=block_tables, out=out, case=case,
                num_seqs=n_seqs)


# --- the two kernel variants -------------------------------------------------
# "tma"      make_tensor_descriptor path. Whether it lowers to a per-iteration
#            tensormap_create (opt OFF) or is demoted to pointers (opt ON) is
#            decided by the compiler gate pass, controlled at the shell level
#            via TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE.
# "pointer"  tensor-of-pointers: explicit tl.load, no descriptor.
CASES = ("tma", "pointer")


def _launch(state):
    """One kernel launch into state['out'], dispatched by state['case']."""
    cfg = state["cfg"]
    kv_cache = state["kv_cache"]
    out = state["out"]
    kernel = paged_kv_load_pointer_kernel if state["case"] == "pointer" \
        else paged_kv_load_kernel
    kernel[(state["num_seqs"], )](
        out,
        kv_cache,
        state["block_tables"],
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        state["block_tables"].stride(0),
        out.stride(0),
        out.stride(1),
        **cfg,
    )
    return out


def _reference(state):
    """
    torch reference: out[seq, j] = kv_cache[block_tables[seq, j],
                                             :TILE_SIZE, KV_HEAD_IDX, :].
    Mirrors the descriptor math: base picks block + kv-head, the loaded tile is
    rows [0:TILE_SIZE] x cols [0:HEAD_SIZE].
    """
    cfg = state["cfg"]
    kv_cache = state["kv_cache"]
    bt = state["block_tables"].to("cpu").long()
    # kv_cache[bt] -> [NUM_SEQS, NUM_KV_BLOCKS, BLOCK_SIZE, KV_HEADS, HEAD_SIZE]
    gathered = kv_cache[bt]
    return gathered[:, :, :cfg["TILE_SIZE"], cfg["KV_HEAD_IDX"], :].contiguous()


def run(case="tma", num_seqs=None, num_kv_blocks=None):
    """JIT-launch the kernel on the local device. Requires a real GPU."""
    state = _make_state(case=case, num_seqs=num_seqs, num_kv_blocks=num_kv_blocks)
    return _launch(state)


def verify_opt_toggle(arch="sm100", num_kv_blocks=None):
    """
    Confirm the loop-recreated-descriptor optimization actually toggles on/off.

    The optimization is a lowering change (identical numerics), so a correctness
    check can't observe it — the signal is in the TTGIR:
      * opt ON  (default)  -> descriptor demoted to pointers: NO tensormap_create
      * opt OFF (disable knob set) -> raw TMA: tensormap_create PRESENT

    Compiles the tma kernel both ways (no GPU needed) and asserts the toggle.
    Returns 0 on PASS, 1 on FAIL. This is what guards the base-vs-opt benchmark:
    if it fails, the pass isn't built / the knob is broken, and the numbers would
    be meaningless.
    """
    disable_env = "TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE"
    prev = os.environ.get(disable_env)

    def _ttgir_has_create(disable):
        if disable:
            os.environ[disable_env] = "1"
        else:
            os.environ.pop(disable_env, None)
        ir = compile_for_target(arch, stop_after="ttgir", num_kv_blocks=num_kv_blocks)
        return "tensormap_create" in ir

    try:
        on_has = _ttgir_has_create(disable=False)  # opt ON  -> expect False
        off_has = _ttgir_has_create(disable=True)  # opt OFF -> expect True
    finally:
        if prev is None:
            os.environ.pop(disable_env, None)
        else:
            os.environ[disable_env] = prev

    ok = (not on_has) and off_has
    if ok:
        print("[smoke:opt-toggle] PASS — pass ON demotes descriptor (no "
              "tensormap_create); pass OFF keeps it", flush=True)
        return 0
    print("[smoke:opt-toggle] FAIL — optimization did not toggle as expected:", flush=True)
    print(f"    opt ON  tensormap_create present: {on_has} (expected False)", flush=True)
    print(f"    opt OFF tensormap_create present: {off_has} (expected True)", flush=True)
    if on_has:
        print(
            "    -> the gate pass did NOT fire on default compile. Is "
            "RewriteLoopTensorDescriptors built into libtriton?", flush=True)
    if not off_has:
        print(
            "    -> the disable knob did NOT preserve the baseline. Is the "
            "make_ttir gate wired to nvidia_disable_loop_td_rewrite?", flush=True)
    return 1


def smoke_test(case="tma", num_seqs=None, num_kv_blocks=None, check_opt=True):
    """
    Smoke gate before benchmarking. Two independent checks:
      1. correctness — kernel output matches the torch reference.
      2. opt-toggle  — the optimization measurably turns on/off in the IR
                       (only meaningful for the 'tma' case; skipped otherwise).
    Returns 0 only if every applicable check passes.
    """
    import torch

    state = _make_state(case=case, num_seqs=num_seqs, num_kv_blocks=num_kv_blocks)
    out = _launch(state)
    torch.cuda.synchronize() if state["device"] == "cuda" else None
    ref = _reference(state).to(out.device)
    try:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        print(f"[smoke:{case}] PASS — output matches torch reference", flush=True)
        rc = 0
    except AssertionError as e:
        print(f"[smoke:{case}] FAIL — output mismatch:\n" + str(e), flush=True)
        rc = 1

    # The opt-toggle check is about the TMA lowering; only run it for the tma
    # kernel (the pointer kernel has no descriptor to toggle).
    if check_opt and case == "tma":
        rc |= verify_opt_toggle(num_kv_blocks=num_kv_blocks)

    return rc


def _time_launches(state, n):
    """Return a list of n per-launch times (ms), each individually synced."""
    import torch
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(n):
        start.record()
        _launch(state)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms
    return times


def _stats(times):
    """median, mean, std, cv(%) for a list of times."""
    n = len(times)
    s = sorted(times)
    median = s[n // 2]
    mean = sum(times) / n
    var = sum((t - mean)**2 for t in times) / n
    std = var**0.5
    cv = (std / mean * 100.0) if mean > 0 else 0.0
    return median, mean, std, cv


def bench(n_warmup=50, n_trials=100, verify=True, case="tma", num_seqs=None, num_kv_blocks=None):
    """Time the kernel with CUDA events. Returns (median_ms, min_ms, max_ms)."""
    import torch

    state = _make_state(case=case, num_seqs=num_seqs, num_kv_blocks=num_kv_blocks)
    device = state["device"]
    if device != "cuda":
        raise RuntimeError("bench requires a CUDA device")

    if verify:
        out = _launch(state)
        torch.cuda.synchronize()
        ref = _reference(state).to(out.device)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        print(f"[bench:{case}] correctness gate PASS", flush=True)

    for _ in range(n_warmup):
        _launch(state)
    torch.cuda.synchronize()

    times = _time_launches(state, n_trials)
    median, mean, std, cv = _stats(times)
    lo, hi = min(times), max(times)
    print(
        f"[bench:{case}] median {median:.5f} ms  min {lo:.5f}  max {hi:.5f}  "
        f"std {std:.5f}  cv {cv:.2f}%  (n={n_trials}, warmup={n_warmup})", flush=True)
    if cv > 3.0:
        print(
            f"[bench:{case}] NOTE: cv {cv:.2f}% is high — results may be "
            "noisy; try `--calibrate` or lock GPU clocks (nvidia-smi -lgc).", flush=True)
    return median, lo, hi


def calibrate(case="tma", burst=500, cv_target=1.0, num_seqs=None, num_kv_blocks=None):
    """
    Run the kernel for a long burst and recommend warmup / trials for stable
    timing. Returns a dict with the recommendation.

    Method:
      * Time `burst` launches back-to-back (no separate warmup).
      * WARMUP: find where the series "settles" — the first index after which
        every subsequent per-launch time is within `settle_tol` of the
        steady-state median (steady-state = median of the last quarter). This
        captures clock ramp / JIT-cache / first-touch effects.
      * TRIALS: from the settled tail, measure coefficient of variation (CV).
        Recommend enough trials that the median is stable; more variance -> more
        trials. `cv_target` (%) is the "good" threshold we report against.
    """
    import torch

    state = _make_state(case=case, num_seqs=num_seqs, num_kv_blocks=num_kv_blocks)
    if state["device"] != "cuda":
        raise RuntimeError("calibrate requires a CUDA device")

    # correctness gate first (never time a wrong kernel)
    out = _launch(state)
    torch.cuda.synchronize()
    ref = _reference(state).to(out.device)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    print(f"[calibrate:{case}] correctness gate PASS", flush=True)

    times = _time_launches(state, burst)

    # steady-state reference = last quarter of the burst
    tail = times[3 * burst // 4:]
    ss_median, ss_mean, ss_std, ss_cv = _stats(tail)

    # WARMUP: first index after which the running values stay within tol
    settle_tol = max(0.02 * ss_median, 3 * ss_std)  # 2% of median or 3 sigma
    warmup_idx = 0
    for i in range(len(times)):
        if all(abs(t - ss_median) <= settle_tol for t in times[i:]):
            warmup_idx = i
            break
    else:
        warmup_idx = len(times)  # never settled
    # round up to a friendly value with headroom
    rec_warmup = max(20, int(_round_up_nice(warmup_idx * 1.5)))

    # TRIALS: scale with observed CV. Std error of the mean ~ cv/sqrt(n);
    # pick n so relative SE is ~0.1% (i.e. n ~ (cv/0.1)^2), clamped sane.
    if ss_cv <= 0:
        rec_trials = 50
    else:
        rec_trials = int(min(2000, max(50, (ss_cv / 0.1)**2)))
    rec_trials = int(_round_up_nice(rec_trials))

    allmed, allmean, allstd, allcv = _stats(times)
    print(f"[calibrate:{case}] burst={burst}", flush=True)
    print(f"  full-burst : median {allmed:.5f}  mean {allmean:.5f}  "
          f"std {allstd:.5f}  cv {allcv:.2f}%", flush=True)
    print(f"  steady tail: median {ss_median:.5f}  mean {ss_mean:.5f}  "
          f"std {ss_std:.5f}  cv {ss_cv:.2f}%", flush=True)
    print(f"  settled after ~{warmup_idx} launches "
          f"(settle tol +-{settle_tol:.5f} ms)", flush=True)
    quality = "good" if ss_cv <= cv_target else \
              ("noisy" if ss_cv <= 3 * cv_target else "very noisy")
    print(f"  steady-state variance: {quality} (cv {ss_cv:.2f}% vs "
          f"target {cv_target:.2f}%)", flush=True)
    print(f"  >>> RECOMMEND: --warmup {rec_warmup} --trials {rec_trials}", flush=True)
    if ss_cv > 3 * cv_target:
        print(
            "  NOTE: high steady-state variance — check GPU clock locking "
            "(nvidia-smi -lgc), thermals, or other GPU load.", flush=True)

    return dict(warmup=rec_warmup, trials=rec_trials, cv=ss_cv, settled_at=warmup_idx, steady_median=ss_median)


def _round_up_nice(x):
    """Round x up to 1/2/5 x 10^k (e.g. 37 -> 50, 120 -> 200)."""
    import math
    if x <= 0:
        return 0
    k = math.floor(math.log10(x))
    base = 10**k
    for m in (1, 2, 5, 10):
        if x <= m * base:
            return int(m * base)
    return int(10 * base)


def _main():
    import argparse

    # Back-compat: no CLI args -> env-var driven compile path (run_blackwell.sh).
    import sys
    if len(sys.argv) == 1:
        arch = os.environ.get("TMA_UPDATE_ARCH", "sm100")
        if os.environ.get("TMA_UPDATE_COMPILE_ONLY", "0") == "1":
            out_dir = os.environ.get("TMA_UPDATE_OUT_DIR") or None
            stop_after = os.environ.get("TMA_UPDATE_STOP_AFTER", "ttgir")
            compile_for_target(arch, stop_after=stop_after, out_dir=out_dir)
            print(f"[paged_kv_load] compiled for {arch}, stopped after "
                  f"{stop_after}", flush=True)
        else:
            run()
        return 0

    p = argparse.ArgumentParser(description="paged_kv_load kernel driver")
    sub = p.add_subparsers(dest="mode", required=True)

    def add_size_flags(sp):
        # Workload-scaling knobs (lift the timing signal above launch/timer floor).
        sp.add_argument(
            "--num-kv-blocks", type=int, default=None, help="loop trip count; scales per-iteration descriptor cost "
            f"(default {DEFAULT_CONSTEXPRS['NUM_KV_BLOCKS']})")
        sp.add_argument("--num-seqs", type=int, default=None, help=f"grid size / CTAs (default {NUM_SEQS})")

    pc = sub.add_parser("compile", help="compile to IR/PTX (no GPU needed)")
    pc.add_argument("--arch", default="sm100")
    pc.add_argument("--stop-after", default="ptx")
    pc.add_argument("--out-dir", default=None)
    pc.add_argument(
        "--num-kv-blocks", type=int, default=None, help="loop trip count baked into the compiled PTX "
        f"(default {DEFAULT_CONSTEXPRS['NUM_KV_BLOCKS']}); must "
        "match the runtime --num-kv-blocks for the 'opt' override")

    ps = sub.add_parser("smoke", help="verify correctness + opt toggles on/off")
    ps.add_argument("--case", choices=CASES, default="tma")
    ps.add_argument("--no-check-opt", action="store_true", help="skip the opt-toggle IR check (correctness only)")
    add_size_flags(ps)

    pb = sub.add_parser("bench", help="CUDA-event timing (+ correctness gate)")
    pb.add_argument("--warmup", type=int, default=50)
    pb.add_argument("--trials", type=int, default=100)
    pb.add_argument("--no-verify", action="store_true")
    pb.add_argument("--case", choices=CASES, default="tma")
    add_size_flags(pb)

    pcal = sub.add_parser("calibrate", help="measure run-to-run variance; recommend warmup/trials")
    pcal.add_argument("--case", choices=CASES, default="tma")
    pcal.add_argument("--burst", type=int, default=500, help="launches to time back-to-back (default 500)")
    pcal.add_argument("--cv-target", type=float, default=1.0, help="'good' steady-state CV%% threshold (default 1.0)")
    add_size_flags(pcal)

    args = p.parse_args()
    if args.mode == "compile":
        compile_for_target(args.arch, stop_after=args.stop_after, out_dir=args.out_dir,
                           num_kv_blocks=args.num_kv_blocks)
        return 0
    if args.mode == "smoke":
        return smoke_test(case=args.case, num_seqs=args.num_seqs, num_kv_blocks=args.num_kv_blocks,
                          check_opt=not args.no_check_opt)
    if args.mode == "bench":
        bench(n_warmup=args.warmup, n_trials=args.trials, verify=not args.no_verify, case=args.case,
              num_seqs=args.num_seqs, num_kv_blocks=args.num_kv_blocks)
        return 0
    if args.mode == "calibrate":
        calibrate(case=args.case, burst=args.burst, cv_target=args.cv_target, num_seqs=args.num_seqs,
                  num_kv_blocks=args.num_kv_blocks)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
