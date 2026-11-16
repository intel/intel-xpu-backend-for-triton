"""#8261 diagnostic on the REAL Inductor split-scan kernel (torch b016e3e, cumsum over 50M fp32, R0_BLOCK=2048, 16 warps).

The kernel body and the look-back are copied verbatim from the generated code / triton_helpers, plus instrumentation:
  arrival[pid] : order in which the block actually started (atomic counter, first instruction)
  pub[pid]     : value of that counter right after the block published its own aggregate
  wait[pid]    : spin iterations of the longest single wait (== BOUND: gave up, kernel continues)
  seen[pid]    : counter value when it gave up;  tgt[pid]: block it was waiting on
Classification of a give-up on target t: arrival[t] > seen  -> t had NOT started while we waited
                                          arrival[t] <= seen -> t had started but had not published.
Launch pattern mimics Inductor autotune benchmarking: bursts of launches, each preceded by a workspace
zero_() and a 256 MB cache-flush fill, one synchronize per burst.
"""
import argparse, json, os, time
import torch, triton, triton.language as tl
from torch._inductor.runtime import triton_helpers

p = argparse.ArgumentParser()
p.add_argument("--bursts", type=int, default=20)
p.add_argument("--per-burst", type=int, default=10)
p.add_argument("--bound", type=int, default=1 << 22)
p.add_argument("--tag", default=f"pid{os.getpid()}")
a = p.parse_args()


@triton.jit
def _add(a, b):
    return a + b


@triton.jit
def lookback_instr(scratch_base, block_value, index, combine_fn, DTYPE_VALUE_AS_UINT: tl.constexpr,
                   DTYPE_PACK: tl.constexpr, ctr, pub, wait, seen, tgt, BOUND: tl.constexpr):
    DTYPE_VALUE = block_value.dtype
    pack = triton_helpers.pack_value_flag(block_value, tl.full(block_value.shape, 1, DTYPE_VALUE_AS_UINT),
                                          DTYPE_VALUE_AS_UINT, DTYPE_PACK)
    if index > 0:
        tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    tl.store(pub + index, tl.atomic_add(ctr, 0, sem="relaxed"))
    exclusive_prefix = tl.zeros([], DTYPE_VALUE)
    prefix_valid = False
    test_target = index - 1
    longest = 0
    while test_target >= 0:
        flag = tl.full([], 0, DTYPE_VALUE_AS_UINT)
        n = 0
        while (flag == 0) & (n < BOUND):
            pack = tl.atomic_add(scratch_base + test_target, 0, sem="relaxed")
            flag = triton_helpers.unpack_flag(pack, DTYPE_VALUE_AS_UINT)
            n += 1
        longest = tl.maximum(longest, n)
        if flag == 0:  # gave up: record and stop looking back (result is wrong, that's fine here)
            tl.store(seen + index, tl.atomic_add(ctr, 0, sem="relaxed"))
            tl.store(tgt + index, test_target)
            test_target = -1
        else:
            value = triton_helpers.unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT)
            if prefix_valid:
                exclusive_prefix = combine_fn(value, exclusive_prefix)
            else:
                exclusive_prefix = value
                prefix_valid = True
            if flag == 2:
                test_target = -1
            else:
                test_target = test_target - 1
    tl.store(wait + index, longest)
    if prefix_valid:
        inclusive_prefix = combine_fn(exclusive_prefix, block_value)
    else:
        inclusive_prefix = block_value
    pack = triton_helpers.pack_value_flag(inclusive_prefix, tl.full([], 2, DTYPE_VALUE_AS_UINT),
                                          DTYPE_VALUE_AS_UINT, DTYPE_PACK)
    tl.atomic_xchg(scratch_base + index, pack, sem="relaxed")
    return exclusive_prefix


@triton.jit
def triton_spl_instr(in_ptr0, out_ptr0, ws_ptr, ctr, arrival, pub, wait, seen, tgt, R0_BLOCK: tl.constexpr,
                     BOUND: tl.constexpr):
    tl.store(arrival + tl.program_id(0), tl.atomic_add(ctr, 1, sem="relaxed"))
    r0_numel = 50000000
    XBLOCK: tl.constexpr = 1
    xoffset = tl.program_id(1) * XBLOCK
    r0_offset = tl.program_id(0) * R0_BLOCK
    r0_index = r0_offset + tl.arange(0, R0_BLOCK)[:]
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    tmp0 = tl.load(in_ptr0 + (r0_index), r0_mask, other=0.0)
    tmp2 = tl.broadcast_to(tmp0.to(tl.float32), [R0_BLOCK])
    tmp3 = tl.reduce(tmp2, 0, _add)
    tmp4 = lookback_instr(ws_ptr.to(tl.pointer_type(tl.uint64)) + xoffset * 1 * tl.num_programs(0), tmp3,
                          tl.program_id(0), _add, DTYPE_VALUE_AS_UINT=tl.uint32, DTYPE_PACK=tl.uint64,
                          ctr=ctr, pub=pub, wait=wait, seen=seen, tgt=tgt, BOUND=BOUND)
    tmp5 = tl.associative_scan(tmp2, 0, _add)
    tmp6 = _add(tmp4, tmp5)
    tmp7 = tl.where(roffset == 0, tmp5, tmp6)
    tl.store(out_ptr0 + (tl.broadcast_to(r0_index, [R0_BLOCK])), tmp7, r0_mask)


dev, N, RB = "xpu", 50_000_000, 2048
G = triton.cdiv(N, RB)
x = torch.rand(N, device=dev); out = torch.empty_like(x)
ws = torch.zeros(1562504, dtype=torch.uint8, device=dev)
flush = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int32, device=dev)
bufs = [{k: torch.zeros(G, dtype=torch.int32, device=dev) for k in ("arrival", "pub", "wait", "seen", "tgt")}
        | {"ctr": torch.zeros(1, dtype=torch.int32, device=dev)} for _ in range(a.per_burst)]
for burst in range(a.bursts):
    t0 = time.time()
    for b in bufs:
        for v in b.values():
            v.zero_()
        b["tgt"].fill_(-1)
    for b in bufs:
        flush.zero_(); ws.zero_()
        triton_spl_instr[(G, 1, 1)](x, out, ws, b["ctr"], b["arrival"], b["pub"], b["wait"], b["seen"], b["tgt"],
                                    R0_BLOCK=RB, BOUND=a.bound, num_warps=16)
    torch.xpu.synchronize(); ms = (time.time() - t0) * 1e3
    gave = ns = sn = 0; ex = None; lag = 0; ooo = 0
    for b in bufs:
        arr, se, tg = b["arrival"].cpu(), b["seen"].cpu(), b["tgt"].cpu()
        lag = max(lag, int((arr[:-1] - arr[1:]).clamp(min=0).max())); ooo += int((arr[:-1] > arr[1:]).sum())
        idx = (tg >= 0).nonzero().flatten()
        for i in idx.tolist():
            t = int(tg[i]); gave += 1
            if arr[t] > se[i]:
                ns += 1
            else:
                sn += 1
            if ex is None:
                ex = {"block": i, "its_start": int(arr[i]), "target": t, "target_start": int(arr[t]),
                      "counter_when_gave_up": int(se[i]), "target_published_at_counter": int(b["pub"].cpu()[t])}
    ref_err = None
    print(json.dumps({"tag": a.tag, "burst": burst, "launches": len(bufs), "ms": round(ms, 1), "gave_up": gave,
                      "target_not_started": ns, "target_started_not_published": sn, "max_start_lag": lag,
                      "out_of_order_pairs_per_launch": ooo // len(bufs), "example": ex}), flush=True)
