# Print the Inductor-generated code for cumsum(50M) on XPU. mode=asis | fakecuda (device check sees "cuda").
import sys, unittest.mock as um, torch
from torch._inductor.utils import run_and_get_code, fresh_inductor_cache
from torch._inductor.codegen import triton_split_scan as tss
from torch._inductor.graph import GraphLowering
torch._inductor.config.compile_threads = 1
if sys.argv[1] == "fakecuda":
    orig = tss.TritonSplitScanKernel.iteration_ranges_get_pid
    def wrapped(self, entry):
        with um.patch.object(GraphLowering, "get_current_device_or_throw", lambda self: torch.device("cuda")):
            return orig(self, entry)
    tss.TritonSplitScanKernel.iteration_ranges_get_pid = wrapped
with fresh_inductor_cache():
    a = torch.rand(50_000_000, device="xpu")
    r, codes = run_and_get_code(torch.compile(lambda a: a.cumsum(-1)), a)
    ref = a.double().cumsum(0)
    print(f"# rel_err {float(((r.double() - ref).abs().max()) / ref.abs().max()):.2e}", file=sys.stderr)
print(codes[0])
