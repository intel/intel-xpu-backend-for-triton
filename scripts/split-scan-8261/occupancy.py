"""Measure how many 16-warp workgroups of the split-scan shape are resident at once on this GPU.
Each block increments a live counter, records the maximum, holds its slot for a while, then decrements.
Uses the same R0_BLOCK=2048 load + reduce as the split-scan kernel so register/SLM use is similar."""
import torch, triton, triton.language as tl


@triton.jit
def occ(x, live, maxlive, dummy, R0_BLOCK: tl.constexpr, HOLD: tl.constexpr):
    pid = tl.program_id(0)
    v = tl.sum(tl.load(x + pid * R0_BLOCK + tl.arange(0, R0_BLOCK)), 0)
    n = tl.atomic_add(live, 1, sem="relaxed") + 1
    tl.atomic_max(maxlive, n, sem="relaxed")
    i = 0
    acc = 0
    while i < HOLD:
        acc += tl.atomic_add(dummy + (pid % 64), 0, sem="relaxed")
        i += 1
    tl.atomic_add(live, -1, sem="relaxed")
    tl.store(x + pid * R0_BLOCK, v + acc.to(tl.float32) * 0)


G, RB = 24415, 2048
x = torch.rand(G * RB, device="xpu")
for warps in (16, 8):
    res = []
    for _ in range(3):
        live = torch.zeros(1, dtype=torch.int32, device="xpu"); mx = torch.zeros_like(live)
        dummy = torch.zeros(64, dtype=torch.int32, device="xpu")
        occ[(G,)](x, live, mx, dummy, R0_BLOCK=RB, HOLD=20000, num_warps=warps)
        torch.xpu.synchronize(); res.append(int(mx.item()))
    p = torch.xpu.get_device_properties()
    print(f"num_warps={warps}: max resident workgroups = {res}  (gpu_subslice_count={getattr(p, 'gpu_subslice_count', '?')}, "
          f"max_compute_units={getattr(p, 'max_compute_units', '?')})", flush=True)
