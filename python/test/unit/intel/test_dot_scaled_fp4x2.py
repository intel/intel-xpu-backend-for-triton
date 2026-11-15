"""tt.dot_scaled with e2m1 operands typed tl.float4e2m1x2 (torch.float4_e2m1fn_x2).

e2m1 operands can be passed packed in pairs either as uint8 or as the native
fp4e2m1x2 type. Both must compile, use the same DPAS sub-group size, and give
the same result bit for bit (issue #8557: the fp4e2m1x2 variant aborted the
compile on an element bit-width assert).
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu
from triton.tools.tensor_descriptor import TensorDescriptor


@triton.jit
def _dot_scaled_ptr(a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, c_ptr, a_valid, M: tl.constexpr, N: tl.constexpr,
                    K: tl.constexpr, A_FMT: tl.constexpr, B_FMT: tl.constexpr, A_MASKED: tl.constexpr):
    A_K: tl.constexpr = K // 2 if A_FMT == "e2m1" else K
    B_K: tl.constexpr = K // 2 if B_FMT == "e2m1" else K
    rm, rn = tl.arange(0, M), tl.arange(0, N)
    rs = tl.arange(0, K // 32)
    a_ptrs = a_ptr + rm[:, None] * A_K + tl.arange(0, A_K)[None, :]
    if A_MASKED:
        # A mask that is uniform over the tile and no `other`: masked-off
        # elements read as zero bits. Without native scaled DPAS the fp4
        # operand keeps a blocked layout, so LowerTo2DBlockLoad does not turn
        # this load into a 2D block load; pointer-load.mlir covers that path.
        a = tl.load(a_ptrs, mask=a_valid > 0)
    else:
        a = tl.load(a_ptrs)
    b = tl.load(b_ptr + tl.arange(0, B_K)[:, None] * N + rn[None, :])
    a_scale = None
    if A_FMT == "e2m1":
        a_scale = tl.load(a_scale_ptr + rm[:, None] * (K // 32) + rs[None, :])
    b_scale = None
    if B_FMT == "e2m1":
        b_scale = tl.load(b_scale_ptr + rn[:, None] * (K // 32) + rs[None, :])
    c = tl.dot_scaled(a, a_scale, A_FMT, b, b_scale, B_FMT)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], c)


@triton.jit
def _dot_scaled_desc_loop_body(a_desc, a_scale_ptr, b_desc, c_ptr, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr):
    pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k in range(0, K, BLOCK_K):
        a = a_desc.load([pid_m * BLOCK_M, k // 2])
        b = b_desc.load([k, pid_n * BLOCK_N])
        rs = k // 32 + tl.arange(0, BLOCK_K // 32)
        a_scale = tl.load(a_scale_ptr + rm[:, None] * (K // 32) + rs[None, :])
        acc = tl.dot_scaled(a, a_scale, "e2m1", b, None, "bf16", acc)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc)


@triton.jit
def _dot_scaled_device_desc_loop(a_ptr, a_scale_ptr, b_ptr, c_ptr, M, N, K, BLOCK_M: tl.constexpr,
                                 BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    a_desc = tl.make_tensor_descriptor(a_ptr, [M, K // 2], [K // 2, 1], [BLOCK_M, BLOCK_K // 2])
    b_desc = tl.make_tensor_descriptor(b_ptr, [K, N], [N, 1], [BLOCK_K, BLOCK_N])
    _dot_scaled_desc_loop_body(a_desc, a_scale_ptr, b_desc, c_ptr, N, K, BLOCK_M, BLOCK_N, BLOCK_K)


@triton.jit
def _dot_scaled_host_desc_loop(a_desc, a_scale_ptr, b_desc, c_ptr, N, K, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                               BLOCK_K: tl.constexpr):
    _dot_scaled_desc_loop_body(a_desc, a_scale_ptr, b_desc, c_ptr, N, K, BLOCK_M, BLOCK_N, BLOCK_K)


@triton.jit
def _copy(x_ptr, y_ptr, num_rows, M: tl.constexpr, N: tl.constexpr, MASK: tl.constexpr):
    rm = tl.arange(0, M)
    offs = rm[:, None] * N + tl.arange(0, N)[None, :]
    if MASK == "row_limit":
        # A row-limit mask, which MaterializeBlockPointer still marks for block IO.
        mask = (rm < num_rows)[:, None]
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)
    elif MASK == "uniform":
        # A mask that is uniform over the tile and no `other`: masked-off
        # elements read as zero bits.
        tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=num_rows > 0))
    else:
        tl.store(y_ptr + offs, tl.load(x_ptr + offs))


def _fp4x2(x, use_fp4x2):
    return x.view(torch.float4_e2m1fn_x2) if use_fp4x2 else x


def _target_arch():
    return triton.runtime.driver.active.get_current_target().arch


def _check_same(run):
    c_u8, kernel_u8 = run(False)
    c_fp4x2, kernel_fp4x2 = run(True)
    assert kernel_fp4x2.metadata.threads_per_warp == kernel_u8.metadata.threads_per_warp
    # Both variants must take the DPAS path wherever the target has DPAS. With a
    # minimum sub-group size of 8 (DG2, ARL-H) DPAS needs 8 threads per warp,
    # which these kernels do not use.
    arch = _target_arch()
    expect_dpas = bool(arch.get("has_subgroup_matrix_multiply_accumulate", False)) and min(
        arch.get("sub_group_sizes") or [16]) >= 16
    assert ("#ttig.dpas<" in kernel_u8.asm["ttgir"]) == expect_dpas
    assert ("#ttig.dpas<" in kernel_fp4x2.asm["ttgir"]) == expect_dpas
    torch.testing.assert_close(c_fp4x2, c_u8, atol=0, rtol=0)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_fmt, b_fmt", [("e2m1", "bf16"), ("bf16", "e2m1"), ("e2m1", "e2m1")])
def test_dot_scaled_fp4x2_matches_uint8(a_fmt, b_fmt, device):
    M = N = K = 128
    generator = torch.Generator(device="cpu").manual_seed(0)

    def operand(fmt, shape_unpacked, pack_dim):
        if fmt == "bf16":
            return torch.randn(shape_unpacked, generator=generator).to(torch.bfloat16).to(device)
        shape = list(shape_unpacked)
        shape[pack_dim] //= 2
        return torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8).to(device)

    a = operand(a_fmt, (M, K), 1)
    b = operand(b_fmt, (K, N), 0)
    a_scale = torch.randint(120, 134, (M, K // 32), generator=generator, dtype=torch.uint8).to(device)
    b_scale = torch.randint(120, 134, (N, K // 32), generator=generator, dtype=torch.uint8).to(device)

    def run(use_fp4x2):
        c = torch.empty((M, N), dtype=torch.float32, device=device)
        kernel = _dot_scaled_ptr[(1, )](_fp4x2(a, use_fp4x2 and a_fmt == "e2m1"), a_scale,
                                        _fp4x2(b, use_fp4x2 and b_fmt == "e2m1"), b_scale, c, 1, M, N, K, a_fmt, b_fmt,
                                        False)
        return c, kernel

    _check_same(run)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("a_valid", [1, 0])
def test_dot_scaled_fp4x2_masked_operand(a_valid, device):
    M = N = K = 128
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), generator=generator, dtype=torch.uint8).to(device)
    b = torch.randn((K, N), generator=generator).to(torch.bfloat16).to(device)
    a_scale = torch.randint(120, 134, (M, K // 32), generator=generator, dtype=torch.uint8).to(device)

    def run(use_fp4x2):
        c = torch.empty((M, N), dtype=torch.float32, device=device)
        kernel = _dot_scaled_ptr[(1, )](_fp4x2(a, use_fp4x2), a_scale, b, None, c, a_valid, M, N, K, "e2m1", "bf16",
                                        True)
        return c, kernel

    c_u8, _ = run(False)
    if a_valid == 0:
        torch.testing.assert_close(c_u8, torch.zeros_like(c_u8), atol=0, rtol=0)
    _check_same(run)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("host_desc", [False, True])
def test_dot_scaled_fp4x2_tensor_descriptor_loop(host_desc, device, with_allocator):
    M, N, K = 256, 256, 512
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
    generator = torch.Generator(device="cpu").manual_seed(0)
    a = torch.randint(0, 256, (M, K // 2), generator=generator, dtype=torch.uint8).to(device)
    a_scale = torch.randint(122, 130, (M, K // 32), generator=generator, dtype=torch.uint8).to(device)
    b = torch.randn((K, N), generator=generator).to(torch.bfloat16).to(device)
    # `with_allocator` restores the null allocator afterwards; its default allocates on cuda.
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device=device))

    def run(use_fp4x2):
        c = torch.empty((M, N), dtype=torch.float32, device=device)
        grid = (M // BLOCK_M, N // BLOCK_N)
        a_op = _fp4x2(a, use_fp4x2)
        if host_desc:
            # Host-side descriptors are rewritten to pointer loads with zero padding.
            a_desc = TensorDescriptor.from_tensor(a_op, [BLOCK_M, BLOCK_K // 2])
            b_desc = TensorDescriptor.from_tensor(b, [BLOCK_K, BLOCK_N])
            kernel = _dot_scaled_host_desc_loop[grid](a_desc, a_scale, b_desc, c, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                      num_stages=3)
        else:
            kernel = _dot_scaled_device_desc_loop[grid](a_op, a_scale, b, c, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K,
                                                        num_stages=3)
        return c, kernel

    _check_same(run)


@pytest.mark.skipif(not is_xpu(), reason="XPU-only test")
@pytest.mark.parametrize("mask, num_rows", [("none", 64), ("row_limit", 40), ("uniform", 64), ("uniform", 0)])
def test_fp4x2_load_store(mask, num_rows, device):
    M = N = 64
    x = torch.randint(0, 256, (M, N), dtype=torch.uint8, device=device)
    # Start from all-ones bytes so that stored zero padding is visible.
    y = torch.full_like(x, 0xFF)
    kernel = _copy[(1, )](_fp4x2(x, True), _fp4x2(y, True), num_rows, M, N, mask)
    if mask != "none" and _target_arch().get("has_2d_block_io", False):
        # MaterializeBlockPointer must still mark the masked load for block
        # IO, which it does only after its mask and pitch checks pass. The
        # store gets the attribute too, so check the load line itself.
        loads = [line for line in kernel.asm["ttgir"].splitlines() if " tt.load " in line]
        assert loads and all("ttig.block_io" in line for line in loads), loads
    expected = x.clone()
    if mask == "row_limit":
        expected[num_rows:] = 0xFF  # masked-off rows are not stored
    elif mask == "uniform" and num_rows == 0:
        expected.zero_()  # masked-off loads read zero bits
    torch.testing.assert_close(y, expected, atol=0, rtol=0)
