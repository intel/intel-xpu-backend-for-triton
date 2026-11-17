"""Standalone reproducer for #8261: Inductor split-scan (decoupled look-back) hang on Intel GPUs.

Each iteration compiles `cumsum` over a long 1-D tensor with a fresh Inductor cache, so Inductor
autotunes the split-scan kernel again (the hang in CI is in that first autotune benchmark), then
checks the result against the ATen (non-Inductor) XPU cumsum computed in float64.

A hang is detected by faulthandler's C-level watchdog (works even if the main thread never
releases the GIL inside zeDeviceSynchronize): it dumps all Python stacks to stderr and exits (code 1).

--no-split forces Scan.num_splits = 1 on XPU, i.e. the single-block scan instead of TritonSplitScanKernel.

  python repro.py --iters 50 --timeout 600 [--numel 50000000] [--dtype float32] [--no-split] [--tag p1-0]
Output: one JSON line per iteration on stdout.
"""

import argparse
import faulthandler
import json
import os
import sys
import time

import torch
from torch._inductor.utils import fresh_inductor_cache

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=20)
p.add_argument("--timeout", type=int, default=600, help="seconds per iteration before declaring a hang")
p.add_argument("--numel", type=int, default=10000 * 5000)  # = test_kernel_benchmark.test_split_scan
p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16", "float64", "int32"])
p.add_argument("--no-split", action="store_true", help="disable split scan for XPU (num_splits=1)")
p.add_argument("--ticket", action="store_true",
               help="prototype fix: split-scan block index from an atomic ticket instead of tl.program_id(0)")
p.add_argument("--tag", default=f"pid{os.getpid()}")
args = p.parse_args()

dev = "xpu"
if args.no_split:
    import torch._inductor.ir as ir

    _orig_num_splits = ir.Scan.num_splits  # bound classmethod

    def _num_splits_no_xpu_split(*a, **kw):
        hint, n = _orig_num_splits(*a, **kw)
        return (hint, 1) if getattr(kw.get("device"), "type", None) == "xpu" else (hint, n)

    ir.Scan.num_splits = staticmethod(_num_splits_no_xpu_split)
if args.ticket:
    # Prototype fix for the forward-progress hazard of decoupled look-back: a workgroup takes its block
    # index from an atomic counter when it starts running, so the predecessor it spins on is always a
    # workgroup that has already started. One counter per x-row lives in the zero-filled workspace.
    # The workspace is not re-zeroed between autotune repetitions, but every launch consumes exactly
    # num_programs(0) tickets on an in-order queue, so `% num_programs(0)` maps each launch to 0..N-1.
    from torch._inductor.codegen.triton_split_scan import TritonSplitScanKernel
    from torch._inductor.utils import sympy_product
    from torch._inductor.codegen.simd import prefix_is_reduction

    _orig_get_pid = TritonSplitScanKernel.iteration_ranges_get_pid

    def _ticket_get_pid(self, entry):
        if not prefix_is_reduction(entry.prefix):
            return _orig_get_pid(self, entry)
        if not getattr(self, "_ticket_emitted", False):
            self._ticket_emitted = True
            rows = sympy_product(n for pfx, n in self.numels.items() if not prefix_is_reduction(pfx))
            ws, _, off = self.args.workspace(nelem=8 * rows, zero_fill=True)  # 8 B per row keeps alignment
            base = f"({ws} + {self.index_to_str(off)})" if off != 0 else ws
            self.body.writeline(
                f"r0_ticket = tl.atomic_add({base}.to(tl.pointer_type(tl.int32)) + 2 * tl.program_id(1), 1, "
                f"sem=\"relaxed\") % tl.num_programs(0)"
            )
        return "r0_ticket"

    TritonSplitScanKernel.iteration_ranges_get_pid = _ticket_get_pid

# Count autotune benchmarking (the CI hang is inside CachingAutotuner.benchmark_all_configs).
import torch._inductor.runtime.triton_heuristics as th
from torch._inductor.codecache import PyCodeCache

_bench = {"calls": 0, "configs": 0, "kernels": set()}
_orig_bac = th.CachingAutotuner.benchmark_all_configs


def _counting_bac(self, *a, **kw):
    _bench["calls"] += 1
    _bench["configs"] += len(self.launchers)
    _bench["kernels"].add(self.fn.__name__)
    return _orig_bac(self, *a, **kw)


th.CachingAutotuner.benchmark_all_configs = _counting_bac
dtype = getattr(torch, args.dtype)
info = dict(tag=args.tag, pid=os.getpid(), torch=torch.__version__, device=torch.xpu.get_device_name(0),
            numel=args.numel, dtype=args.dtype, split=not args.no_split, ticket=args.ticket)
print(json.dumps({"event": "start", **info}), flush=True)

for it in range(args.iters):
    faulthandler.dump_traceback_later(args.timeout, exit=True, file=sys.stderr)
    t0 = time.time()
    _bench.update(calls=0, configs=0, kernels=set())
    PyCodeCache.cache_clear(purge=True)  # don't reuse an already-autotuned module from memory
    with fresh_inductor_cache():
        torch._dynamo.reset()
        f = torch.compile(lambda a: a.cumsum(-1))
        if args.dtype == "int32":
            a = torch.randint(-3, 4, (args.numel,), device=dev, dtype=dtype)
        else:
            a = torch.rand(args.numel, device=dev, dtype=dtype)
        out = f(a)
        torch.xpu.synchronize()
    dt = time.time() - t0
    faulthandler.cancel_dump_traceback_later()
    ref = a.double().cumsum(-1)
    err = (out.double() - ref).abs().max().item() / max(1.0, ref.abs().max().item())
    print(json.dumps({"event": "iter", "tag": args.tag, "it": it, "sec": round(dt, 2), "rel_err": err,
                      "autotune_calls": _bench["calls"], "configs_benchmarked": _bench["configs"],
                      "kernels": sorted(_bench["kernels"])}), flush=True)

print(json.dumps({"event": "done", "tag": args.tag}), flush=True)
