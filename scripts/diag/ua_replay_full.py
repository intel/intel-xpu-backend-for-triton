"""Faithful local replay of unified_attention_benchmark.py (#8742): providers triton -> pytorch -> sycl-tla per config,
verification against the XPU torch reference (the OLD behaviour), 100 warmup + 10 timed calls per provider.

usage: replay_full.py [model_index ...]      (MODELS_BF16 indices, default 0 1)
Set PYTHONPATH to a directory containing another vllm_xpu_kernels to A/B kernel versions.
"""
import ast
import sys
import time
from itertools import product

import numpy as np
import torch
import triton

BENCH = str(__import__("pathlib").Path(__file__).resolve().parents[2] / "benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention_benchmark.py")
from vllm.v1.attention.ops.triton_unified_attention import unified_attention  # noqa: E402
import vllm_xpu_kernels  # noqa: E402
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func as sycl_tla_attention  # noqa: E402

SEQ_LENS_LIGHT = [[(4096, 4096)], [(1, k) for k in [1513, 4100, 530, 123, 4803, 434, 3015, 34]],
                  [(1, k) for k in [7168, 8192, 12288, 16384]]]
SEQ_LENS_HEAVY = [[(7168, 7168)], [(2048, 2048)] * 4, [(8192, 8192)] * 2,
                  [(1, k) for k in ([1513, 4100, 530, 123, 4803, 434, 3015, 34, 256, 1024, 768, 2048, 192, 384, 1280, 96] * 16)[:248]] +
                  [(256, 256)] * 8]
MODELS = [(32, 8, 128, None), (64, 8, 128, None), (64, 8, 128, 8192), (28, 4, 128, None), (64, 4, 128, None),
          (64, 4, 128, 256)]


def load_ref():
    tree = ast.parse(open(BENCH).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ref_paged_attn")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), BENCH, "exec"), ns)  # noqa: S102 - trusted repo code
    return ns["ref_paged_attn"]


def enough_memory(qh, kh, hs, sw, seq, nb, bs, safety=0.80):
    d = 2
    qlens = [s[0] for s in seq]
    kvlens = [s[1] for s in seq]
    query_mem = sum(qlens) * qh * hs * d
    kv_mem = 2 * nb * bs * kh * hs * d
    bt_mem = len(seq) * ((max(kvlens) + bs - 1) // bs) * 4
    ref = 0
    for ql, kl in seq:
        rep = 2 * kl * qh * hs * d if qh != kh else 0
        attn = qh * ql * kl * 4
        ref = max(ref, rep + 2 * attn + ql * kl * 4 + ql * kl + (ql * kl if sw is not None else 0))
    total = max(query_mem + kv_mem + bt_mem + ref, query_mem + kv_mem + 2 * query_mem + bt_mem + 8 * len(seq))
    return total < torch.xpu.get_device_properties(0).total_memory * safety


def configs(models):
    heavy = {id(s) for s in SEQ_LENS_HEAVY}
    for mi in models:
        qh, kh, hs, sw = MODELS[mi]
        for seq, nb, bs in product(SEQ_LENS_LIGHT + SEQ_LENS_HEAVY, [32768, 2048], [16, 64]):
            if id(seq) in heavy and bs != 64:
                continue
            yield mi, qh, kh, hs, sw, seq, nb, bs


def bench_loop(fn, warmup=100, reps=10):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    for _ in range(reps):
        fn()
    torch.xpu.synchronize()


def call_provider(provider, ref_paged_attn, qh, kh, hs, sw, seq, nb, bs):
    dtype = torch.bfloat16
    torch.manual_seed(20)
    torch.set_default_device("xpu")
    qlens = [x[0] for x in seq]
    kvlens = [x[1] for x in seq]
    nseq = len(seq)
    window = (sw - 1, 0) if sw is not None else (-1, -1)
    scale = hs**-0.5
    query = torch.randn(sum(qlens), qh, hs, dtype=dtype)
    kc = torch.randn(nb, bs, kh, hs, dtype=dtype)
    vc = torch.randn_like(kc)
    mb = (max(kvlens) + bs - 1) // bs
    bt = torch.randint(0, nb, (nseq, mb), dtype=torch.int32)

    def torch_fn():
        return ref_paged_attn(query=query, key_cache=kc, value_cache=vc, query_lens=qlens, kv_lens=kvlens,
                              block_tables=bt, scale=scale, sliding_window=sw, soft_cap=None)

    if provider == "pytorch":
        bench_loop(torch_fn)
        return "n/a (timed only)"
    expected = torch_fn()
    cu = torch.tensor([0] + qlens, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    kvt = torch.tensor(kvlens, dtype=torch.int32)
    if provider == "triton":
        out = torch.empty_like(query)
        thr, nseg = 32, 16
        so = torch.empty(thr, qh, nseg, triton.next_power_of_2(hs), dtype=torch.float32)
        sm = torch.empty(thr, qh, nseg, dtype=torch.float32)
        se = torch.empty(thr, qh, nseg, dtype=torch.float32)

        def fn():
            unified_attention(q=query, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=kvt, max_seqlen_q=max(qlens),
                              max_seqlen_k=max(kvlens), softmax_scale=scale, causal=True, window_size=window,
                              block_table=bt, softcap=0, q_descale=None, k_descale=None, v_descale=None,
                              seq_threshold_3D=thr, num_par_softmax_segments=nseg, softmax_segm_output=so,
                              softmax_segm_max=sm, softmax_segm_expsum=se, use_td=False)
            return out
    else:

        def fn():
            return sycl_tla_attention(q=query, k=kc, v=vc, max_seqlen_q=max(qlens), cu_seqlens_q=cu,
                                      max_seqlen_k=max(kvlens), seqused_k=kvt, softmax_scale=scale, causal=True,
                                      window_size=list(window), block_table=bt, softcap=0)

    got = fn()
    torch.xpu.synchronize()
    a, d = got.float().cpu().numpy(), expected.float().cpu().numpy()
    bad = ~np.isclose(a, d, atol=2.5e-2, rtol=1e-2, equal_nan=True)
    msg = f"verify mismatches {int(bad.sum())}/{bad.size}"
    if bad.any():
        torch.set_default_device("cpu")
        r32 = ref_paged_attn(query=query.float().cpu(), key_cache=kc.float().cpu(), value_cache=vc.float().cpu(),
                             query_lens=qlens, kv_lens=kvlens, block_tables=bt.cpu(), scale=scale,
                             sliding_window=sw, soft_cap=None).float().numpy()
        msg += (f"; vs cpu32: provider {int((~np.isclose(a, r32, atol=2.5e-2, rtol=1e-2, equal_nan=True)).sum())}, "
                f"xpu-reference {int((~np.isclose(d, r32, atol=2.5e-2, rtol=1e-2, equal_nan=True)).sum())}")
        torch.set_default_device("xpu")
    bench_loop(fn)
    return msg


def main():
    models = [int(x) for x in sys.argv[1:]] or [0, 1]
    ref_paged_attn = load_ref()
    print(f"torch {torch.__version__} triton {triton.__version__} vllm_xpu_kernels from {vllm_xpu_kernels.__file__}",
          flush=True)
    for mi, qh, kh, hs, sw, seq, nb, bs in configs(models):
        tag = f"m{mi} {qh}-{kh}-{hs}-sw{sw} seq[{len(seq)}x{seq[0]}] nb{nb} bs{bs}"
        if not enough_memory(qh, kh, hs, sw, seq, nb, bs):
            print(f"{tag}: SKIP memory", flush=True)
            continue
        for provider in ("triton", "pytorch", "sycl-tla"):
            t = time.time()
            try:
                res = call_provider(provider, ref_paged_attn, qh, kh, hs, sw, seq, nb, bs)
            except Exception as e:  # pylint: disable=broad-except
                res = f"ERROR {type(e).__name__}: {str(e)[:80]}"
            print(f"{tag} [{provider}]: {res} ({time.time() - t:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
