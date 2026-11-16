"""Diagnostics for https://github.com/intel/intel-xpu-backend-for-triton/issues/8742 (experiment, not for merge).

Failing config: 64-8-128, 4 x (2048, 2048), num_blocks=2048, block_size=64, bf16, seed 20 (inputs as in the benchmark).
Runs the benchmark's ref_paged_attn on the XPU several times and compares with a float32 CPU reference, then checks the
reference stage by stage (QK einsum, in-place and out-of-place softmax, PV einsum) for sequence 0.
usage: ua_phase1.py N_RUNS [--imports mod1,mod2] [--ops-only]
"""
import argparse
import ast
import importlib
import pathlib

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention_benchmark.py"


def load_ref():
    tree = ast.parse(BENCH.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "ref_paged_attn")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), str(BENCH), "exec"), ns)  # noqa: S102 - trusted repo code
    return ns["ref_paged_attn"]


def bad_mask(a, b, atol, rtol):
    return ~torch.isclose(a.float().cpu(), b.float().cpu(), atol=atol, rtol=rtol, equal_nan=True)


def describe(name, mask, head_dim=0):
    n = int(mask.sum())
    msg = f"{name}: {n} / {mask.numel()} bad"
    if n:
        per_head = mask.flatten(1).any(dim=1) if head_dim == 0 else mask.transpose(0, head_dim).flatten(1).any(dim=1)
        heads = per_head.nonzero().flatten().tolist()
        msg += f"; bad heads ({len(heads)}): {heads[:12]}"
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=int)
    ap.add_argument("--imports", default="")
    ap.add_argument("--ops-only", action="store_true")
    args = ap.parse_args()
    for m in filter(None, args.imports.split(",")):
        importlib.import_module(m)
        print("imported", m, flush=True)

    qh, kh, hs, nb, bs = 64, 8, 128, 2048, 64
    seq = [(2048, 2048)] * 4
    ql, kl = [s[0] for s in seq], [s[1] for s in seq]
    scale = hs**-0.5
    torch.manual_seed(20)
    torch.set_default_device("xpu")
    q = torch.randn(sum(ql), qh, hs, dtype=torch.bfloat16)
    kc = torch.randn(nb, bs, kh, hs, dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    bt = torch.randint(0, nb, (len(seq), 32), dtype=torch.int32)
    kw = dict(query_lens=ql, kv_lens=kl, block_tables=bt, scale=scale, sliding_window=None, soft_cap=None)

    idx = bt[0].cpu().tolist()
    q0 = q[:2048] * scale
    k0 = torch.repeat_interleave(kc[idx].view(-1, kh, hs)[:2048], qh // kh, dim=1)
    v0 = torch.repeat_interleave(vc[idx].view(-1, kh, hs)[:2048], qh // kh, dim=1)

    if args.ops_only:  # only the two einsums, for ONEDNN_VERBOSE
        a = torch.einsum("qhd,khd->hqk", q0, k0).float()
        torch.xpu.synchronize()
        p = torch.softmax(a.masked_fill(torch.triu(torch.ones(2048, 2048), diagonal=1).bool(), float("-inf")), dim=-1)
        e = torch.einsum("hqk,khd->qhd", p.to(torch.bfloat16), v0)
        torch.xpu.synchronize()
        return

    cfg = [ln.strip() for ln in torch.__config__.show().splitlines() if "MKL-DNN" in ln]
    free, total = torch.xpu.mem_get_info()
    print(f"torch {torch.__version__} | {cfg} | {torch.xpu.get_device_name(0)} | free {free / 2**30:.1f}/{total / 2**30:.1f} GiB",
          flush=True)
    ref = load_ref()
    with torch.device("cpu"):
        cpu = ref(query=q.float().cpu(), key_cache=kc.float().cpu(), value_cache=vc.float().cpu(),
                  **{**kw, "block_tables": bt.cpu()})
    for i in range(args.runs):
        out = ref(query=q.clone(), key_cache=kc, value_cache=vc, **kw)
        torch.xpu.synchronize()
        m = bad_mask(out, cpu, 2.5e-2, 1e-2)
        d01 = float((out[0, 0].float() - out[0, 1].float()).abs().max())
        print(f"run {i}: XPU reference vs fp32 CPU: {int(m.sum())} / {m.numel()} mismatched; token0 head0-vs-head1 diff {d01:.4f}",
              flush=True)
    # stage by stage, sequence 0
    a_x = torch.einsum("qhd,khd->hqk", q0, k0).float()
    a_c = torch.einsum("qhd,khd->hqk", q0.float().cpu(), k0.float().cpu())
    describe("stage A QK einsum XPU vs CPU32", bad_mask(a_x, a_c, 3e-2, 3e-2))
    msk = torch.triu(torch.ones(2048, 2048), diagonal=1).bool()
    b_x = a_x.masked_fill(msk, float("-inf"))
    s_in = b_x.clone()
    torch.softmax(s_in, dim=-1, out=s_in)
    s_out = torch.softmax(b_x, dim=-1)
    s_cpu = torch.softmax(b_x.cpu(), dim=-1)
    describe("stage C softmax IN-PLACE XPU vs CPU", bad_mask(s_in, s_cpu, 1e-5, 1e-3))
    describe("stage C' softmax OUT-OF-PLACE XPU vs CPU", bad_mask(s_out, s_cpu, 1e-5, 1e-3))
    p = s_cpu.to(torch.bfloat16).to("xpu")
    e_x = torch.einsum("hqk,khd->qhd", p, v0)
    e_c = torch.einsum("hqk,khd->qhd", p.float().cpu(), v0.float().cpu())
    describe("stage E PV einsum XPU vs CPU32", bad_mask(e_x, e_c, 3e-2, 3e-2), head_dim=1)


if __name__ == "__main__":
    main()
