# SPDX-License-Identifier: Apache-2.0
"""Standalone correctness sweep for the in-kernel-sentinel 3D unified attention.

Run against a vLLM checkout that has ``unified_attention.patch`` applied (the
``triton_unified_attention`` module must import from that tree)::

    PYTHONPATH=<patched-vllm-tree> python test_sentinel_correctness.py
    PYTHONPATH=<patched-vllm-tree> python test_sentinel_correctness.py --reused

The first form sweeps both legal tile sizes (TILE_SIZE in {16, 32} for bf16;
32 only for fp8) over KV lengths around the segment-layout boundaries and
compares the 3D decode output against a CPU reference.  ``--reused`` exercises
the serving-path case of persistent segment buffers reused (and pre-poisoned)
across calls.  Requires an Intel XPU device.


Validates that the 3D decode path produces correct output for BOTH legal tile
sizes (TILE_SIZE in {16, 32}) -- the case the old code could not autotune
because reduce_segments recomputed its segment mask from TILE_SIZE.

For each (config) we force the producer autotuner to a single TILE_SIZE,
sweep KV lengths around the segment-layout boundaries
cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE), and compare the 3D kernel
output against a CPU reference (ref_paged_attn) AND against the 2D path.
"""

import sys
import torch
import triton

from vllm.v1.attention.ops import triton_unified_attention as M
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEVICE = "xpu"
NUM_SEGMENTS = 16  # NUM_SEGMENTS_PER_SEQ used by the 3D path
FP8_DTYPE = torch.float8_e4m3fn


def ref_paged_attn(query, key_cache, value_cache, query_lens, kv_lens,
                   block_tables, scale, sliding_window=None, soft_cap=None):
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape
    outputs = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx:start_idx + query_len]
        q = q * scale
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]
        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sw = torch.triu(empty_mask,
                            diagonal=kv_len - (query_len + sliding_window) + 1
                            ).bool().logical_not()
            mask |= sw
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start_idx += query_len
    return torch.cat(outputs, dim=0)


def force_tile_size(ts):
    """Pin the producer autotuner to a single TILE_SIZE and clear its cache."""
    M.kernel_unified_attention.configs = [
        triton.Config({"TILE_SIZE": ts}, num_stages=2)
    ]
    M.kernel_unified_attention.cache.clear()


def run_once(kv_lens, *, num_heads=(8, 2), head_size=128, block_size=64,
             num_blocks=2048, dtype=torch.bfloat16, q_dtype=None,
             sliding_window=None, soft_cap=None, seq_threshold_3D=8, seed=0):
    """One decode batch (query_len==1 per seq). Returns (out, ref, used_3d)."""
    torch.manual_seed(seed)
    num_seqs = len(kv_lens)
    query_lens = [1] * num_seqs
    nqh, nkvh = num_heads
    max_query_len = 1
    max_kv_len = max(kv_lens)
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size ** -0.5

    query = torch.randn(sum(query_lens), nqh, head_size, dtype=dtype, device=DEVICE)
    key_cache = torch.randn(num_blocks, block_size, nkvh, head_size, dtype=dtype, device=DEVICE)
    value_cache = torch.randn_like(key_cache)
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device=DEVICE).cumsum(0, dtype=torch.int32)
    kv_t = torch.tensor(kv_lens, dtype=torch.int32, device=DEVICE)
    max_blocks = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0, num_blocks, (num_seqs, max_blocks), dtype=torch.int32, device=DEVICE)

    out_dtype = dtype
    qd = kd = vd = None
    q_in, k_in, v_in = query, key_cache, value_cache
    if q_dtype is not None:
        q_in = query.to(q_dtype)
        k_in = key_cache.to(q_dtype)
        v_in = value_cache.to(q_dtype)
        kd = torch.rand((num_seqs, nkvh), dtype=torch.float32, device=DEVICE)
        vd = torch.rand((num_seqs, nkvh), dtype=torch.float32, device=DEVICE)

    output = torch.empty(sum(query_lens), nqh, head_size, dtype=out_dtype, device=DEVICE)

    hsp = triton.next_power_of_2(head_size)
    segm_out = torch.empty((seq_threshold_3D, nqh, NUM_SEGMENTS, hsp), dtype=torch.float32, device=DEVICE)
    segm_max = torch.empty((seq_threshold_3D, nqh, NUM_SEGMENTS), dtype=torch.float32, device=DEVICE)
    segm_exp = torch.empty((seq_threshold_3D, nqh, NUM_SEGMENTS), dtype=torch.float32, device=DEVICE)

    use_3d = (max_query_len == 1) and (num_seqs <= seq_threshold_3D) and (sliding_window is None)

    unified_attention(
        q=q_in, k=k_in, v=v_in, out=output,
        cu_seqlens_q=cu_q, seqused_k=kv_t,
        max_seqlen_q=max_query_len, max_seqlen_k=max_kv_len,
        softmax_scale=scale, causal=True, window_size=window_size,
        block_table=block_tables, softcap=soft_cap if soft_cap is not None else 0,
        q_descale=qd, k_descale=kd, v_descale=vd,
        seq_threshold_3D=seq_threshold_3D, num_par_softmax_segments=NUM_SEGMENTS,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max, softmax_segm_expsum=segm_exp,
    )

    ref = ref_paged_attn(query, key_cache, value_cache, query_lens, kv_lens,
                         block_tables, scale, sliding_window, soft_cap)
    return output, ref, use_3d


def main():
    torch.set_default_device(DEVICE)
    # KV lengths chosen around the segment-layout boundaries:
    #   tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS * TILE_SIZE)
    # For NUM_SEGMENTS=16 and TILE_SIZE in {16,32}, boundaries cluster at
    # multiples of 256 and 512; also small lens that under-fill segments.
    KV_LENS = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
               255, 256, 257, 511, 512, 513, 1023, 1024, 1025,
               4095, 4096, 4097]
    HEAD_SIZES = [128, 256]
    QDTYPES = [None, FP8_DTYPE]

    fails = []
    total = 0
    print(f"{'cfg':<22}{'tile':<6}{'used_3d':<9}{'kv_len':<8}{'max_abs_err':<14}status")
    print("-" * 70)
    for q_dtype in QDTYPES:
        # fp8 path only supports TILE_SIZE=32 (pruned otherwise); bf16 tries both.
        tiles = [32] if q_dtype is not None else [16, 32]
        for head_size in HEAD_SIZES:
            for tile in tiles:
                force_tile_size(tile)
                for kv in KV_LENS:
                    total += 1
                    tag = f"{'fp8' if q_dtype else 'bf16'}/hs{head_size}"
                    try:
                        out, ref, used_3d = run_once(
                            [kv], head_size=head_size, q_dtype=q_dtype)
                        out_f = out.to(torch.float32).cpu()
                        ref_f = ref.to(torch.float32).cpu()
                        err = (out_f - ref_f).abs().max().item()
                        nan = torch.isnan(out_f).any().item()
                        atol = 2e-1 if q_dtype is not None else 2e-2
                        ok = (not nan) and (err <= atol)
                        status = "OK" if ok else ("NAN" if nan else "FAIL")
                        if not ok:
                            fails.append((tag, tile, kv, err, nan))
                        print(f"{tag:<22}{tile:<6}{str(used_3d):<9}{kv:<8}{err:<14.5f}{status}")
                    except Exception as e:
                        fails.append((tag, tile, kv, "EXC", repr(e)))
                        print(f"{tag:<22}{tile:<6}{'?':<9}{kv:<8}{'-':<14}EXC {e!r}")
    print("-" * 70)
    print(f"total={total} fails={len(fails)}")
    if fails:
        print("\nFAILURES:")
        for f in fails:
            print("  ", f)
        sys.exit(1)
    print("ALL PASS")


def main_reused_buffers():
    """Serving-path scenario: persistent segm_* buffers reused across calls.

    Pre-poison the buffers with finite garbage and reuse them across calls with
    shrinking KV lengths.  A short KV (few live segments) following a long KV
    (many live segments) is exactly where stale finite values in now-unwritten
    slots would corrupt the result IF the reducer trusted them.  The in-kernel
    sentinel makes the producer overwrite every slot (live data or -inf) each
    call, so this must stay correct.
    """
    torch.set_default_device(DEVICE)
    NS = NUM_SEGMENTS
    hs, (nqh, nkvh), bs, nb = 128, (8, 2), 64, 2048
    hsp = triton.next_power_of_2(hs)
    scale = hs ** -0.5
    seqthr = 8
    force_tile_size(16)

    # persistent buffers, poisoned with large finite values
    segm_out = torch.full((seqthr, nqh, NS, hsp), 7.0, dtype=torch.float32)
    segm_max = torch.full((seqthr, nqh, NS), 123.0, dtype=torch.float32)
    segm_exp = torch.full((seqthr, nqh, NS), 99.0, dtype=torch.float32)

    def call(kv, seed):
        torch.manual_seed(seed)
        q = torch.randn(1, nqh, hs, dtype=torch.bfloat16)
        kc = torch.randn(nb, bs, nkvh, hs, dtype=torch.bfloat16)
        vc = torch.randn_like(kc)
        cu = torch.tensor([0, 1], dtype=torch.int32)
        kvt = torch.tensor([kv], dtype=torch.int32)
        mb = (kv + bs - 1) // bs
        bt = torch.randint(0, nb, (1, mb), dtype=torch.int32)
        out = torch.empty(1, nqh, hs, dtype=torch.bfloat16)
        unified_attention(
            q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=kvt,
            max_seqlen_q=1, max_seqlen_k=kv, softmax_scale=scale, causal=True,
            window_size=(-1, -1), block_table=bt, softcap=0,
            q_descale=None, k_descale=None, v_descale=None,
            seq_threshold_3D=seqthr, num_par_softmax_segments=NS,
            softmax_segm_output=segm_out, softmax_segm_max=segm_max,
            softmax_segm_expsum=segm_exp)
        ref = ref_paged_attn(q, kc, vc, [1], [kv], bt, scale, None, None)
        err = (out.float().cpu() - ref.float().cpu()).abs().max().item()
        nan = torch.isnan(out.float()).any().item()
        return err, nan

    print("Reused-buffer test (persistent buffers poisoned to finite 7/123/99):")
    fails = 0
    for kv, seed in [(4096, 1), (64, 2), (2048, 3), (16, 4), (257, 5), (1, 6)]:
        err, nan = call(kv, seed)
        ok = err < 2e-2 and not nan
        fails += not ok
        print(f"  kv={kv:<6} max_abs_err={err:.5f} nan={nan} {'OK' if ok else 'FAIL'}")
    if fails:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--reused", action="store_true",
                   help="run the reused-persistent-buffer scenario instead of the sweep")
    a = p.parse_args()
    if a.reused:
        main_reused_buffers()
    else:
        main()
