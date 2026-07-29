# pylint: skip-file
"""
unified_attention_nv.py — NVIDIA driver for the vLLM unified-attention kernel.

Companion to paged_kv_load.py. Same three cases, same output format, but the
kernel under test is the REAL vLLM `unified_attention` rather than a synthetic
reproducer:

    base   TMA, optimization OFF   (TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1)
    tofp   no descriptors at all   (use_td=False -> upstream's masked tl.load)
    opt    TMA, optimization ON    (default Hopper+ path; the gate pass demotes
                                    loop-recreated descriptors to pointers)

`base` and `opt` compile the SAME source with the pass toggled — that is the A/B.
`tofp` is upstream's own pointer path, the hand-written reference point.

Unlike the XPU benchmark this is derived from, the kernel is imported, never
vendored: vLLM must be installed. See RUN.md ('NVIDIA vLLM install') for the
recipe (`install-vllm.sh` hardcodes VLLM_TARGET_DEVICE=xpu and cannot be used).

Descriptor coverage is controlled by --td-mode:
    kv    (default) use_td=True only          -> in-loop K/V descriptors
    all   use_td=True + use_td_qo=True        -> also out-of-loop Q/O

`--td-mode all` is the configuration that exercises SELECTIVITY: Q/O are
hoistable so the pass must keep them on TMA while demoting K/V. Under `kv` the
pass demotes everything and tensormap_create goes to zero, which only proves the
pass fires. Whether the pin accepts `use_td_qo` as a kwarg is checked at runtime
and reported, not assumed.

Subcommands
-----------
    probe      What does the installed vLLM support? (no GPU)
    ir         Compile-only: count tensormap_create per case (no GPU)
    smoke      Launch once, verify vs torch reference (GPU)
    bench      Time the cases, one table (GPU)
"""

import os
import sys
from typing import Optional

# ---------------------------------------------------------------- config

# One decode-heavy shape. Per-iteration descriptor cost scales with the KV loop
# trip count, so long kv_lens with query_len 1 is where the signal lives.
# (q_heads, k_heads, head_size)
MODEL = (32, 8, 128)
SEQ_LENS = [(1, k) for k in (7168, 8192, 12288, 16384)]
NUM_BLOCKS = 2048
BLOCK_SIZE = 64

CASES = ("base", "tofp", "opt")
TD_MODES = ("kv", "all")

DISABLE_KNOB = "TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE"


def _case_env(case):
    """Environment overlay for a case. `opt`/`tofp` use the default pass path."""
    return {DISABLE_KNOB: "1"} if case == "base" else {}


def _case_use_td(case, td_mode):
    """(use_td, use_td_qo) for a case."""
    if case == "tofp":
        return False, False
    return True, td_mode == "all"


# ---------------------------------------------------------------- imports

def _import_unified_attention():
    """
    Import vLLM's unified_attention, with an actionable message on failure.

    Both module paths are tried because the symbol moved between vLLM versions;
    the XPU benchmark does the same.
    """
    last = None
    for mod in ("vllm.attention.ops.triton_unified_attention",
                "vllm.v1.attention.ops.triton_unified_attention"):
        try:
            return getattr(__import__(mod, fromlist=["unified_attention"]),
                           "unified_attention")
        except ImportError as e:
            last = e
    raise RuntimeError(
        "cannot import unified_attention from vLLM. This driver does not vendor "
        "the kernel — vLLM must be installed against the pin in "
        "scripts/vllm/vllm-pin.txt. See RUN.md ('NVIDIA vLLM install'); note "
        "scripts/vllm/install-vllm.sh is XPU-only and will not work here."
    ) from last


def _set_allocator():
    """
    Register the global-scratch allocator that on-device TMA descriptors need.

    Hopper `tensormap_create` writes its descriptor into global scratch. Nothing
    in the XPU benchmark path does this — XPU demotes every descriptor in
    make_ttir so scratch is never allocated — so on NVIDIA the launch fails
    without it unless vLLM registers one itself. Registering here is harmless if
    it does.
    """
    import torch
    import triton

    setter = getattr(triton, "set_allocator", None)
    if setter is None:
        raise RuntimeError(
            f"triton.set_allocator not found (triton "
            f"{getattr(triton, '__version__', '?')} at {triton.__file__}). "
            "This is almost certainly torch's bundled Triton wheel shadowing "
            "your source build; reinstall with `pip install -e .` from your "
            "triton checkout, torch FIRST and editable triton LAST.")

    def _alloc(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    setter(_alloc)


# ---------------------------------------------------------------- probe

def probe():
    """
    Report what the installed vLLM supports, so a later failure is not a mystery.

    Checks, in order: the kernel imports; `use_td` is accepted; `use_td_qo` is
    accepted (decides whether --td-mode all is available); nothing gates the
    descriptor path on the current platform. Returns 0 if `use_td` works, since
    that alone is enough for --td-mode kv.
    """
    import inspect

    try:
        fn = _import_unified_attention()
    except RuntimeError as e:
        print(f"[probe] FAIL {e}", flush=True)
        return 1

    print(f"[probe] unified_attention from {inspect.getmodule(fn).__name__}",
          flush=True)

    params = inspect.signature(fn).parameters
    has_td = "use_td" in params
    has_qo = "use_td_qo" in params
    print(f"[probe] use_td accepted:    {has_td}", flush=True)
    print(f"[probe] use_td_qo accepted: {has_qo}", flush=True)

    # A platform gate on the descriptor path would silently give us the pointer
    # kernel in all three cases, making base == opt == tofp. Grep the source
    # rather than infer it: cheap, and the failure it prevents is a whole
    # benchmark run of meaningless numbers.
    try:
        src = inspect.getsource(inspect.getmodule(fn))
        for marker in ("current_platform", "is_xpu", "is_cuda"):
            if marker in src and "USE_TD" in src:
                print(f"[probe] NOTE: '{marker}' appears in the kernel module; "
                      "confirm the descriptor path is not gated off on CUDA",
                      flush=True)
                break
    except OSError:
        pass

    if not has_td:
        print("[probe] FAIL: no use_td parameter — this pin predates the "
              "descriptor toggle, so there is nothing to A/B.", flush=True)
        return 1
    if not has_qo:
        print("[probe] --td-mode all unavailable (no use_td_qo); only in-loop "
              "K/V descriptors can be enabled, so the pass will demote ALL of "
              "them and tensormap_create will drop to zero. That tests that the "
              "pass fires, not that it is selective.", flush=True)
    print("[probe] OK", flush=True)
    return 0


# ---------------------------------------------------------------- workload

def _make_state(seed=20):
    """
    Build the tensors unified_attention() needs. Mirrors the XPU benchmark's
    setup so the two are comparable.
    """
    import torch

    torch.manual_seed(seed)
    q_heads, k_heads, head_size = MODEL
    dtype = torch.bfloat16
    dev = "cuda"

    query_lens = [q for q, _ in SEQ_LENS]
    kv_lens = [k for _, k in SEQ_LENS]
    num_seqs = len(SEQ_LENS)
    max_kv_len = max(kv_lens)
    blocks_per_seq = (max_kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE

    query = torch.randn(sum(query_lens), q_heads, head_size, dtype=dtype, device=dev)
    key_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, k_heads, head_size, dtype=dtype, device=dev)
    value_cache = torch.randn_like(key_cache)

    return {
        "q": query,
        "k": key_cache,
        "v": value_cache,
        "out": torch.empty_like(query),
        "cu_seqlens_q": torch.tensor([0] + query_lens, dtype=torch.int32,
                                     device=dev).cumsum(dim=0, dtype=torch.int32),
        "seqused_k": torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        # block_tables maps logical block -> arbitrary physical block. This
        # indirection is the whole point: the descriptor base derives from an
        # in-loop tl.load of this table, which is impure and therefore not
        # hoistable, which is why the descriptor must be built per iteration.
        "block_table": torch.randint(0, NUM_BLOCKS, (num_seqs, blocks_per_seq),
                                     dtype=torch.int32, device=dev),
        "max_seqlen_q": max(query_lens),
        "max_seqlen_k": max_kv_len,
        "scale": head_size**-0.5,
        "query_lens": query_lens,
        "kv_lens": kv_lens,
    }


def _launch(state, case, td_mode):
    """Call unified_attention() for one case. Returns the output tensor."""
    fn = _import_unified_attention()
    use_td, use_td_qo = _case_use_td(case, td_mode)

    kwargs = dict(
        q=state["q"], k=state["k"], v=state["v"], out=state["out"],
        cu_seqlens_q=state["cu_seqlens_q"], seqused_k=state["seqused_k"],
        max_seqlen_q=state["max_seqlen_q"], max_seqlen_k=state["max_seqlen_k"],
        softmax_scale=state["scale"], causal=True, window_size=(-1, -1),
        block_table=state["block_table"], softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        use_td=use_td,
    )
    if use_td_qo:
        kwargs["use_td_qo"] = True

    fn(**kwargs)
    return state["out"]


def _reference(state):
    """Torch reference. Pure torch, lifted from the XPU benchmark."""
    import torch

    query, key_cache, value_cache = state["q"], state["k"], state["v"]
    block_tables = state["block_table"].cpu().numpy()
    _, block_size, k_heads, head_size = key_cache.shape
    scale = state["scale"]

    outputs = []
    start = 0
    for i, (qlen, kvlen) in enumerate(zip(state["query_lens"], state["kv_lens"])):
        q = query[start:start + qlen] * scale
        idx = block_tables[i, :(kvlen + block_size - 1) // block_size]
        k = key_cache[idx].view(-1, k_heads, head_size)[:kvlen]
        v = value_cache[idx].view(-1, k_heads, head_size)[:kvlen]
        if q.shape[1] != k.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, rep, dim=1)
            v = torch.repeat_interleave(v, rep, dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        mask = torch.triu(torch.ones(qlen, kvlen, device=q.device),
                          diagonal=kvlen - qlen + 1).bool()
        attn.masked_fill_(mask, float("-inf"))
        torch.softmax(attn, dim=-1, out=attn)
        outputs.append(torch.einsum("hqk,khd->qhd", attn.to(v.dtype), v))
        start += qlen
    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------- ir

def _count_tensormap_create(case, td_mode):
    """
    Compile the kernel for one case and count `tensormap_create` in its TTGIR.

    A count, not a boolean: with Q/O descriptors enabled the correct result is a
    DROP, not zero — the hoistable ones must survive. Uses a compilation
    listener to read the IR that was actually produced, so the pipeline is the
    real one rather than a hand-rolled reconstruction.

    Returns (count, n_kernels). Requires a GPU because launching is what triggers
    JIT compilation; the count itself is a compile-time property.
    """
    import torch
    import triton
    from triton import knobs

    counts = []

    def listener(*, src, metadata, metadata_group, times, cache_hit):
        for name, path in metadata_group.items():
            if name.endswith("ttgir"):
                try:
                    counts.append(open(path).read().count("tensormap_create"))
                except OSError:
                    pass

    prev_listener = knobs.compilation.listener
    prev_always = os.environ.get("TRITON_ALWAYS_COMPILE")
    prev_disable = os.environ.get(DISABLE_KNOB)

    # Without ALWAYS_COMPILE a cache hit skips the pipeline and the listener
    # sees no IR — and base/opt differ only by a knob the cache key may not
    # cover, so a stale entry would silently make them identical.
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    for key, val in _case_env(case).items():
        os.environ[key] = val
    if case != "base":
        os.environ.pop(DISABLE_KNOB, None)

    try:
        knobs.compilation.listener = listener
        _set_allocator()
        _launch(_make_state(), case, td_mode)
        torch.cuda.synchronize()
    finally:
        knobs.compilation.listener = prev_listener
        for key in _case_env(case):
            os.environ.pop(key, None)
        if prev_always is None:
            os.environ.pop("TRITON_ALWAYS_COMPILE", None)
        else:
            os.environ["TRITON_ALWAYS_COMPILE"] = prev_always
        if prev_disable is not None:
            os.environ[DISABLE_KNOB] = prev_disable

    return sum(counts), len(counts)


def ir(td_mode="kv"):
    """
    Compare tensormap_create counts across cases — the compile-side answer to
    "does the optimization apply, and is it selective?"

    Expected, --td-mode all: base > opt > 0. The surviving ones are Q/O
    (hoistable, kept on TMA); the difference is K/V (loop-recreated, demoted).
    Expected, --td-mode kv:  base > 0, opt == 0. Every descriptor is
    loop-recreated, so all of them demote.
    """
    print(f"[ir] td-mode={td_mode}", flush=True)
    results = {}
    for case in ("base", "opt"):
        count, n = _count_tensormap_create(case, td_mode)
        results[case] = count
        print(f"[ir] {case:5s} tensormap_create = {count:3d}  ({n} kernel(s))",
              flush=True)

    base, opt = results["base"], results["opt"]
    if base == 0:
        print("[ir] FAIL: baseline has no tensormap_create — the TMA path never "
              "ran. Either use_td did not reach the kernel, or the GPU is "
              "pre-Hopper (sm<90 demotes every descriptor in make_ttir).",
              flush=True)
        return 1
    if opt == base:
        print("[ir] FAIL: the pass changed nothing. It may not be built into "
              "libtriton, or no descriptor here is classified loop-recreated.",
              flush=True)
        return 1

    print(f"[ir] PASS: {base} -> {opt} ({base - opt} descriptor(s) demoted)",
          flush=True)
    if td_mode == "all" and opt > 0:
        print(f"[ir] SELECTIVE: {opt} descriptor(s) kept on TMA — the hoistable "
              "Q/O ones. This is the case that tests per-descriptor "
              "selectivity, not just that the pass fires.", flush=True)
    elif td_mode == "all":
        print("[ir] NOTE: opt reached 0 with Q/O descriptors enabled. Either "
              "use_td_qo did not take effect, or Q/O were demoted too — worth "
              "reading the TTGIR before trusting the bench numbers.", flush=True)
    return 0


# ---------------------------------------------------------------- smoke

def smoke(case="opt", td_mode="kv"):
    """Launch once and verify against the torch reference."""
    import torch

    _set_allocator()
    for key, val in _case_env(case).items():
        os.environ[key] = val
    try:
        out = _launch(_make_state(), case, td_mode)
        torch.cuda.synchronize()
    finally:
        for key in _case_env(case):
            os.environ.pop(key, None)

    ref = _reference(_make_state()).to(out.device)
    try:
        torch.testing.assert_close(out, ref, atol=2.5e-2, rtol=1e-2)
        print(f"[smoke:{case}] PASS — output matches torch reference", flush=True)
        return 0
    except AssertionError as e:
        print(f"[smoke:{case}] FAIL — output mismatch:\n{e}", flush=True)
        return 1


# ---------------------------------------------------------------- bench

def _stats(times):
    """median, mean, std, cv(%). Same shape as paged_kv_load._stats."""
    n = len(times)
    median = sorted(times)[n // 2]
    mean = sum(times) / n
    std = (sum((t - mean)**2 for t in times) / n)**0.5
    return median, mean, std, (std / mean * 100.0 if mean > 0 else 0.0)


def bench_case(case, td_mode="kv", n_warmup=50, n_trials=100, verify=True):
    """
    Time one case with CUDA events, behind a correctness gate.

    The gate matters more here than in paged_kv_load: base/opt/tofp are three
    different lowerings of the same math, so a demotion bug shows up as wrong
    numerics, and an ungated timing run would report it as a speedup.
    """
    import torch

    _set_allocator()
    for key, val in _case_env(case).items():
        os.environ[key] = val
    try:
        state = _make_state()

        if verify:
            out = _launch(state, case, td_mode)
            torch.cuda.synchronize()
            ref = _reference(state).to(out.device)
            torch.testing.assert_close(out, ref, atol=2.5e-2, rtol=1e-2)
            print(f"[bench:{case}] correctness gate PASS", flush=True)

        for _ in range(n_warmup):
            _launch(state, case, td_mode)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(n_trials):
            start.record()
            _launch(state, case, td_mode)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    finally:
        for key in _case_env(case):
            os.environ.pop(key, None)

    median, mean, std, cv = _stats(times)
    print(f"[bench:{case}] median {median:.5f} ms  min {min(times):.5f}  "
          f"max {max(times):.5f}  std {std:.5f}  cv {cv:.2f}%  "
          f"(n={n_trials}, warmup={n_warmup})", flush=True)
    if cv > 3.0:
        print(f"[bench:{case}] NOTE: cv {cv:.2f}% is high — try locking clocks "
              "(nvidia-smi -lgc).", flush=True)
    return median


def bench(case="all", td_mode="kv", n_warmup=50, n_trials=100, verify=True):
    """Time the requested cases and print a comparison table."""
    wanted = CASES if case == "all" else (case,)
    medians = {}
    for name in wanted:
        print(f"\n=========== {name} ===========", flush=True)
        try:
            medians[name] = bench_case(name, td_mode, n_warmup, n_trials, verify)
        except Exception as e:  # keep going: one broken case should not hide the rest
            print(f"[bench:{name}] ERROR {type(e).__name__}: {e}", flush=True)

    if len(medians) > 1:
        print(f"\n================= SUMMARY (median ms, td-mode={td_mode}) "
              "=================", flush=True)
        for name in CASES:
            if name in medians:
                print(f"  {name:<6} {medians[name]:>10.5f} ms", flush=True)
        print(flush=True)

        def ratio(a, b, note):
            if a in medians and b in medians and medians[b] > 0:
                r = medians[a] / medians[b]
                delta = (medians[a] - medians[b]) / medians[a] * 100.0
                print(f"  {a} vs {b}: {r:.3f}x ({delta:+.1f}%)  {note}", flush=True)

        ratio("base", "opt", "does the optimization help the TMA path?")
        ratio("base", "tofp", "baseline TMA vs upstream's pointer path")
        ratio("opt", "tofp", "optimized TMA vs pointers (expect ~parity)")

    return 0 if medians else 1


# ---------------------------------------------------------------- cli

def main():
    import argparse

    p = argparse.ArgumentParser(
        description="NVIDIA driver for the vLLM unified-attention kernel")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_td_mode(parser):
        parser.add_argument("--td-mode", choices=TD_MODES, default="kv",
                            help="kv: in-loop K/V descriptors only (default). "
                                 "all: also out-of-loop Q/O — the selectivity test.")

    sub.add_parser("probe", help="report what the installed vLLM supports")

    add_td_mode(sub.add_parser("ir", help="count tensormap_create per case"))

    ps = sub.add_parser("smoke", help="launch once, verify vs torch")
    ps.add_argument("--case", choices=CASES, default="opt")
    add_td_mode(ps)

    pb = sub.add_parser("bench", help="time the cases")
    pb.add_argument("--case", choices=("all",) + CASES, default="all")
    pb.add_argument("--warmup", type=int, default=50)
    pb.add_argument("--trials", type=int, default=100)
    pb.add_argument("--no-verify", action="store_true")
    add_td_mode(pb)

    args = p.parse_args()

    if args.cmd == "probe":
        return probe()
    if args.cmd == "ir":
        return ir(args.td_mode)
    if args.cmd == "smoke":
        return smoke(args.case, args.td_mode)
    return bench(args.case, args.td_mode, args.warmup, args.trials,
                 not args.no_verify)


if __name__ == "__main__":
    sys.exit(main())
