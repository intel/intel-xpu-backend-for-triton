import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.backends.intel import compiler as intel_compiler


def _has_16_wide_dpas():
    if not is_xpu():
        return False
    arch = triton.runtime.driver.active.get_current_target().arch
    return arch["has_subgroup_matrix_multiply_accumulate"] and min(arch["sub_group_sizes"]) == 16


@triton.jit
def chained_dot_kernel(a_ptr, b_ptr, c_ptr, out_ptr, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                       D: tl.constexpr):
    # out = (a @ b) @ c: the first dot feeds the A operand of the second, a chain along A.
    offs_m = tl.arange(0, M)
    offs_k = tl.arange(0, K)
    offs_n = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
    c = tl.load(c_ptr + offs_n[:, None] * D + offs_d[None, :])
    p = tl.dot(a, b).to(tl.float16)
    out = tl.dot(p, c)
    tl.store(out_ptr + offs_m[:, None] * D + offs_d[None, :], out)


@pytest.mark.skipif(not _has_16_wide_dpas(), reason="Chained dot split needs 16-wide DPAS")
@pytest.mark.parametrize("spill_size, warps_per_cta, num_llir_builds", [
    (0, "[8, 1]", 1),
    (4 * intel_compiler.REBUILD_SPILL_BYTES_PER_THREAD, "[4, 2]", 2),
])
def test_chained_dot_split_follows_spill(spill_size, warps_per_cta, num_llir_builds, device, fresh_triton_cache,
                                         monkeypatch):
    # With 8 warps all on M, each warp would hold the whole 128x64 B of the first dot, so AccelerateMatmul marks the
    # module and the backend builds it once to see whether it spills. Pin the spill size the trial build reports.
    monkeypatch.setattr(intel_compiler, "extract_spill_size_from_zebin", lambda _: spill_size)
    llir_builds = []
    make_llir = intel_compiler.XPUBackend.make_llir

    def counting_make_llir(cls, src, metadata, options):
        llir_builds.append(src)
        return make_llir(src, metadata, options)

    monkeypatch.setattr(intel_compiler.XPUBackend, "make_llir", classmethod(counting_make_llir))

    M, K, N, D = 64, 128, 64, 128
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=torch.float16) * 0.1
    b = torch.randn((K, N), device=device, dtype=torch.float16) * 0.1
    c = torch.randn((N, D), device=device, dtype=torch.float16) * 0.1
    out = torch.empty((M, D), device=device, dtype=torch.float32)
    # The split is decided on a native-code build, the path inductor compiles on; drop kernels built by other cases.
    chained_dot_kernel.device_caches[triton.runtime.driver.active.get_current_device()][0].clear()
    kernel = chained_dot_kernel[(1, )](a, b, c, out, M, K, N, D, num_warps=8, generate_native_code=True)

    # A trial that does not spill is reused by the later stages instead of being lowered again.
    assert len(llir_builds) == num_llir_builds
    assert f"warpsPerCTA = {warps_per_cta}" in kernel.asm["ttgir"]
    # The kept TTGIR is the marked module itself; the split rebuild is not marked again.
    assert ("chained_dot_split_candidate" in kernel.asm["ttgir"]) == (spill_size == 0)
    ref = (a.float() @ b.float()).half().float() @ c.float()
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-3)
