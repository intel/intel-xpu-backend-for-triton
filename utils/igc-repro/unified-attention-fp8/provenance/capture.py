"""Capture the SPIR-V + launch parameters of the vLLM unified-attention kernels for one benchmark case.
Usage: capture.py <case-name> <outdir>.  Run with the repo's triton + a vLLM checkout on PYTHONPATH."""
import os, sys, json, shutil
import torch, triton
from triton.runtime.jit import JITFunction

CASES = {
  # name: (seq_lens, sliding_window, num_blocks, block_size)   fp8, q_heads=64, k_heads=8, head_size=128
  "prefill4096": ([(4096, 4096)], None, 2048, 64),
  "decode8":     ([(1, 1513), (1, 4100), (1, 530), (1, 123), (1, 4803), (1, 434), (1, 3015), (1, 34)], None, 2048, 64),
}
name, outdir = sys.argv[1], sys.argv[2]
seq_lens, sw, nb, bs = CASES[name]
QH, KH, HS = 64, 8, 128
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
torch.manual_seed(0); torch.set_default_device("xpu")
ql = [s[0] for s in seq_lens]; kl = [s[1] for s in seq_lens]; ns = len(seq_lens)
q = torch.randn(sum(ql), QH, HS).to(torch.float8_e4m3fn)
kc = torch.randn(nb, bs, KH, HS).to(torch.float8_e4m3fn); vc = torch.randn(nb, bs, KH, HS).to(torch.float8_e4m3fn)
bt = torch.randint(0, nb, (ns, (max(kl) + bs - 1) // bs), dtype=torch.int32)
cu = torch.tensor([0] + ql, dtype=torch.int32).cumsum(0, dtype=torch.int32); kvl = torch.tensor(kl, dtype=torch.int32)
out = torch.empty(q.shape, dtype=torch.bfloat16)
kd = torch.rand((ns, KH), dtype=torch.float32); vd = torch.rand((ns, KH), dtype=torch.float32)
T3, P = 32, 16
segs = dict(softmax_segm_output=torch.empty(T3, QH, P, triton.next_power_of_2(HS), dtype=torch.float32),
            softmax_segm_max=torch.empty(T3, QH, P, dtype=torch.float32), softmax_segm_expsum=torch.empty(T3, QH, P, dtype=torch.float32))

launches = []
orig = JITFunction.run
def hooked(self, *args, **kwargs):
    r = orig(self, *args, **kwargs)
    launches.append((self.fn.__name__, args, kwargs, r)); return r
JITFunction.run = hooked
def call():
    unified_attention(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, seqused_k=kvl, max_seqlen_q=max(ql), max_seqlen_k=max(kl),
        softmax_scale=HS ** -0.5, causal=True, window_size=((sw - 1, 0) if sw else (-1, -1)), block_table=bt, softcap=0,
        q_descale=None, k_descale=kd, v_descale=vd, seq_threshold_3D=T3, num_par_softmax_segments=P, **segs)
call(); torch.xpu.synchronize()
os.makedirs(outdir, exist_ok=True)
tensors = {id(t): n for n, t in dict(q=q, k=kc, v=vc, out=out, block_table=bt, cu_seqlens_q=cu, seqused_k=kvl, k_descale=kd, v_descale=vd,
           segm_output=segs["softmax_segm_output"], segm_max=segs["softmax_segm_max"], segm_expsum=segs["softmax_segm_expsum"]).items()}
rec = []
for i, (kname, args, kw, kern) in enumerate(launches):
    sig = kern.src.signature; md = kern.metadata
    spv = kern.asm.get("spv") or kern.asm.get("spvbin")
    fn = f"{i}_{kname}.spv"; open(os.path.join(outdir, fn), "wb").write(spv if isinstance(spv, bytes) else spv.encode() if isinstance(spv, str) else bytes(spv))
    info = dict(kernel=kname, spv=fn, signature={k: str(v) for k, v in sig.items()}, grid=list(kw.get("grid", ())) if "grid" in kw else None,
                num_warps=md.num_warps, threads_per_warp=getattr(md, "threads_per_warp", 16), build_flags=getattr(md, "build_flags", ""),
                shared=md.shared, global_scratch_size=getattr(md, "global_scratch_size", 0), metadata={k: str(v) for k, v in md._asdict().items()} if hasattr(md, "_asdict") else str(md))
    argvals = {k: v for k, v in kw.items() if k not in ("grid", "warmup", "num_warps", "num_stages", "num_ctas", "maxnreg", "launch_cooperative_grid", "launch_pdl")}
    args_rec = []
    for pname, ptype in sig.items():
        v = argvals.get(pname)
        if str(ptype) == "constexpr": continue
        if hasattr(v, "data_ptr"):
            args_rec.append(dict(name=pname, type=str(ptype), kind="tensor", tensor=tensors.get(id(v), "?"), dtype=str(v.dtype), shape=list(v.shape)))
        elif v is None: args_rec.append(dict(name=pname, type=str(ptype), kind="null"))
        else: args_rec.append(dict(name=pname, type=str(ptype), kind="scalar", value=v))
    info["args"] = args_rec; info["launch_kwargs"] = {k: kw[k] for k in ("num_warps", "num_stages") if k in kw}
    rec.append(info)
for tn, t in dict(block_table=bt, cu_seqlens_q=cu, seqused_k=kvl, k_descale=kd, v_descale=vd).items():
    t.cpu().numpy().tofile(os.path.join(outdir, tn + ".bin"))
json.dump(dict(case=name, seq_lens=seq_lens, sliding_window=sw, num_blocks=nb, block_size=bs, q_heads=QH, k_heads=KH, head_size=HS, launches=rec),
          open(os.path.join(outdir, "capture.json"), "w"), indent=1, default=str)
print("launches:", [(l[0], l[3].metadata.num_warps) for l in launches])
