"""End-to-end correctness test for the LoopDistribute pass
(`tritonintelgpu-loop-distribute`, gated by the `loop_distribute` XPUOptions
field / TRITON_INTEL_ENABLE_LOOP_DISTRIBUTION).

The pass splits an scf.for loop containing exactly two tt.dot ops (sharing
operand A) into two independent loops, each computing a single dot, to
reduce register pressure. It is off by default.

This test compiles and runs two kernel shapes with `loop_distribute` both
disabled and enabled and checks that:
  * both variants produce numerically correct results (vs. a torch reference)
  * both variants agree with each other
  * the pass actually fired when enabled, by counting `scf.for` occurrences
    in the compiled TTIR (2 when enabled, 1 when disabled)

Kernel A is a raw-pointer GEMM whose A/B operands are advanced every
iteration via loop-carried `tt.addptr` (never re-derived from the loop
index) -- the regression shape for the loop-carried-iter_arg-replication fix.
Kernel B is a `tl.make_tensor_descriptor`-based GEMM mirroring the
already-supported fast path used by `fused_gemm_swiglu_kernel` in
benchmarks/triton_kernels_benchmark/fused_gemm_benchmark.py.
Kernel C adds a loop-carried value that reads one dot's accumulator, which
exercises the per-iter_arg ownership path: the chain is computed only in the
loop owning that accumulator, its slot is frozen in the other loop, and its
result is wired from the owner.

The last two tests cover the `loop_distribute_cost_model` XPUOptions field /
TRITON_INTEL_ENABLE_LOOP_DISTRIBUTION_COST_MODEL, which runs the pass only on
loops whose accumulators do not fit the register budget.
"""

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import is_xpu


def _has_2d_block_io():
    """Check if current device supports 2D block I/O."""
    return triton.runtime.driver.active.get_current_target().arch.get('has_2d_block_io', False)


@triton.jit
def _dual_dot_ptr_kernel(
    a_ptr,
    wg_ptr,
    wfc_ptr,
    cg_ptr,
    cfc_ptr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wgk: tl.constexpr,
    stride_wgn: tl.constexpr,
    stride_wfck: tl.constexpr,
    stride_wfcn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    wg_ptrs = wg_ptr + offs_k[:, None] * stride_wgk + offs_n[None, :] * stride_wgn
    wfc_ptrs = wfc_ptr + offs_k[:, None] * stride_wfck + offs_n[None, :] * stride_wfcn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_fc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        w_g = tl.load(wg_ptrs)
        w_fc = tl.load(wfc_ptrs)
        acc_g = tl.dot(a, w_g, acc_g, input_precision="ieee")
        acc_fc = tl.dot(a, w_fc, acc_fc, input_precision="ieee")
        # Loop-carried pointers: advanced every iteration, never re-derived
        # from the loop index.
        a_ptrs += BLOCK_K * stride_ak
        wg_ptrs += BLOCK_K * stride_wgk
        wfc_ptrs += BLOCK_K * stride_wfck

    cg_ptrs = cg_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    cfc_ptrs = cfc_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(cg_ptrs, acc_g.to(cg_ptr.dtype.element_ty))
    tl.store(cfc_ptrs, acc_fc.to(cfc_ptr.dtype.element_ty))


@triton.jit
def _dual_dot_desc_kernel(
    x_ptr,
    wg_ptr,
    wfc_ptr,
    yg_ptr,
    yfc_ptr,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    desc_x = tl.make_tensor_descriptor(x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
    desc_wg = tl.make_tensor_descriptor(wg_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])
    desc_wfc = tl.make_tensor_descriptor(wfc_ptr, shape=[K, N], strides=[N, 1], block_shape=[BLOCK_K, BLOCK_N])

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_fc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = desc_x.load([0, k])
        w_g = desc_wg.load([k, 0])
        w_fc = desc_wfc.load([k, 0])
        acc_g = tl.dot(x, w_g, acc_g)
        acc_fc = tl.dot(x, w_fc, acc_fc)

    desc_yg = tl.make_tensor_descriptor(yg_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_yfc = tl.make_tensor_descriptor(yfc_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N])
    desc_yg.store([0, 0], acc_g.to(yg_ptr.type.element_ty))
    desc_yfc.store([0, 0], acc_fc.to(yfc_ptr.type.element_ty))


@triton.jit
def _dual_dot_carry_reads_acc_kernel(
    a_ptr,
    wg_ptr,
    wfc_ptr,
    cg_ptr,
    cfc_ptr,
    carry_ptr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_wgk: tl.constexpr,
    stride_wgn: tl.constexpr,
    stride_wfck: tl.constexpr,
    stride_wfcn: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    wg_ptrs = wg_ptr + offs_k[:, None] * stride_wgk + offs_n[None, :] * stride_wgn
    wfc_ptrs = wfc_ptr + offs_k[:, None] * stride_wfck + offs_n[None, :] * stride_wfcn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_fc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    carry = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        w_g = tl.load(wg_ptrs)
        w_fc = tl.load(wfc_ptrs)
        # Read acc_g BEFORE the tl.dot below rebinds it, so this reads the
        # accumulator's *iter_arg* (its value at the top of the iteration) and
        # not the dot's result. That makes `carry` a loop-carried chain owned by
        # acc_g's distributed loop.
        carry = carry + acc_g
        acc_g = tl.dot(a, w_g, acc_g, input_precision="ieee")
        acc_fc = tl.dot(a, w_fc, acc_fc, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        wg_ptrs += BLOCK_K * stride_wgk
        wfc_ptrs += BLOCK_K * stride_wfck

    cg_ptrs = cg_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    cfc_ptrs = cfc_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    carry_ptrs = carry_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(cg_ptrs, acc_g.to(cg_ptr.dtype.element_ty))
    tl.store(cfc_ptrs, acc_fc.to(cfc_ptr.dtype.element_ty))
    tl.store(carry_ptrs, carry)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_dual_dot_ptr_gemm_loop_distribute(device):
    """Regression test for the loop-carried-iter_arg-replication fix: A, W_g
    and W_fc pointers are all loop-carried (advanced via `+=`, i.e. tt.addptr
    chains) rather than re-derived from the loop index every iteration."""
    M, N, K, BLOCK_K = 128, 128, 256, 64

    torch.manual_seed(17)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    w_g = torch.randn((K, N), dtype=torch.float16, device=device)
    w_fc = torch.randn((K, N), dtype=torch.float16, device=device)

    ref_g = torch.matmul(a.float(), w_g.float())
    ref_fc = torch.matmul(a.float(), w_fc.float())

    results = {}
    for loop_distribute in (False, True):
        c_g = torch.empty((M, N), dtype=torch.float16, device=device)
        c_fc = torch.empty((M, N), dtype=torch.float16, device=device)

        kernel = _dual_dot_ptr_kernel[(1, 1)](
            a,
            w_g,
            w_fc,
            c_g,
            c_fc,
            K,
            a.stride(0),
            a.stride(1),
            w_g.stride(0),
            w_g.stride(1),
            w_fc.stride(0),
            w_fc.stride(1),
            c_g.stride(0),
            c_g.stride(1),
            BLOCK_M=M,
            BLOCK_N=N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            loop_distribute=loop_distribute,
        )

        ttir = kernel.asm["ttir"]
        expected_loops = 2 if loop_distribute else 1
        assert ttir.count("scf.for") == expected_loops, (
            f"loop_distribute={loop_distribute}: expected {expected_loops} scf.for in ttir, "
            f"got {ttir.count('scf.for')}\n{ttir}")

        err_g = (c_g.float() - ref_g).abs().max().item()
        err_fc = (c_fc.float() - ref_fc).abs().max().item()
        assert err_g < 2e-1, f"loop_distribute={loop_distribute}: acc_g max abs error {err_g} exceeds 2e-1"
        assert err_fc < 2e-1, f"loop_distribute={loop_distribute}: acc_fc max abs error {err_fc} exceeds 2e-1"

        results[loop_distribute] = (c_g.clone(), c_fc.clone())

    # loop_distribute=False and loop_distribute=True must agree with each other.
    torch.testing.assert_close(results[False][0].float(), results[True][0].float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(results[False][1].float(), results[True][1].float(), rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.xfail(not _has_2d_block_io(), reason="2D block I/O not supported", run=False)
def test_dual_dot_desc_gemm_loop_distribute(device):
    """Descriptor-based dual-dot GEMM (mirrors fused_gemm_swiglu_kernel's loop
    body) -- guards that the already-supported descriptor fast path stays
    correct with `loop_distribute` enabled."""
    M, N, K, BLOCK_K = 128, 128, 256, 64

    torch.manual_seed(17)
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    w_g = torch.randn((K, N), dtype=torch.bfloat16, device=device)
    w_fc = torch.randn((K, N), dtype=torch.bfloat16, device=device)

    ref_g = torch.matmul(x.float(), w_g.float())
    ref_fc = torch.matmul(x.float(), w_fc.float())

    results = {}
    for loop_distribute in (False, True):
        y_g = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        y_fc = torch.empty((M, N), dtype=torch.bfloat16, device=device)

        kernel = _dual_dot_desc_kernel[(1, )](
            x,
            w_g,
            w_fc,
            y_g,
            y_fc,
            M,
            N,
            K,
            BLOCK_M=M,
            BLOCK_N=N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            loop_distribute=loop_distribute,
        )

        ttir = kernel.asm["ttir"]
        expected_loops = 2 if loop_distribute else 1
        assert ttir.count("scf.for") == expected_loops, (
            f"loop_distribute={loop_distribute}: expected {expected_loops} scf.for in ttir, "
            f"got {ttir.count('scf.for')}\n{ttir}")

        err_g = (y_g.float() - ref_g).abs().max().item()
        err_fc = (y_fc.float() - ref_fc).abs().max().item()
        # bf16 inputs accumulated over K=256 in fp32; generous absolute
        # tolerance to accommodate bf16 rounding of the inputs.
        assert err_g < 3.0, f"loop_distribute={loop_distribute}: acc_g max abs error {err_g} exceeds 3.0"
        assert err_fc < 3.0, f"loop_distribute={loop_distribute}: acc_fc max abs error {err_fc} exceeds 3.0"

        results[loop_distribute] = (y_g.clone(), y_fc.clone())

    # loop_distribute=False and loop_distribute=True must agree with each other.
    torch.testing.assert_close(results[False][0].float(), results[True][0].float(), rtol=1e-2, atol=5e-2)
    torch.testing.assert_close(results[False][1].float(), results[True][1].float(), rtol=1e-2, atol=5e-2)


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_dual_dot_carry_reads_acc_loop_distribute(device):
    """A loop-carried, non-accumulator value (`carry`) reads acc_g's iter_arg, so
    it can only be computed in the loop that keeps acc_g live.

    This exercises per-iter_arg ownership end to end: `carry`'s chain is cloned
    into acc_g's loop only, its slot is frozen (yielded unchanged) in acc_fc's
    loop, and the returned value is taken from the owning loop. Getting any of
    those three wrong changes `carry` numerically -- it would come out as the
    frozen init (all zeros) or be accumulated twice -- while acc_g and acc_fc
    stay correct, so `carry` is the value that actually discriminates here.
    """
    M, N, K, BLOCK_K = 128, 128, 256, 64

    torch.manual_seed(17)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    w_g = torch.randn((K, N), dtype=torch.float16, device=device)
    w_fc = torch.randn((K, N), dtype=torch.float16, device=device)

    ref_g = torch.matmul(a.float(), w_g.float())
    ref_fc = torch.matmul(a.float(), w_fc.float())
    # carry accumulates acc_g as seen at the TOP of each iteration, i.e. the sum
    # of the partial products over all but the last k-block, prefix by prefix.
    ref_carry = torch.zeros((M, N), dtype=torch.float32, device=device)
    running = torch.zeros((M, N), dtype=torch.float32, device=device)
    for k in range(0, K, BLOCK_K):
        ref_carry += running
        running = running + torch.matmul(a[:, k:k + BLOCK_K].float(), w_g[k:k + BLOCK_K, :].float())

    results = {}
    for loop_distribute in (False, True):
        c_g = torch.empty((M, N), dtype=torch.float16, device=device)
        c_fc = torch.empty((M, N), dtype=torch.float16, device=device)
        c_carry = torch.empty((M, N), dtype=torch.float32, device=device)

        kernel = _dual_dot_carry_reads_acc_kernel[(1, 1)](
            a,
            w_g,
            w_fc,
            c_g,
            c_fc,
            c_carry,
            K,
            a.stride(0),
            a.stride(1),
            w_g.stride(0),
            w_g.stride(1),
            w_fc.stride(0),
            w_fc.stride(1),
            c_g.stride(0),
            c_g.stride(1),
            BLOCK_M=M,
            BLOCK_N=N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            loop_distribute=loop_distribute,
        )

        ttir = kernel.asm["ttir"]
        expected_loops = 2 if loop_distribute else 1
        assert ttir.count("scf.for") == expected_loops, (
            f"loop_distribute={loop_distribute}: expected {expected_loops} scf.for in ttir, "
            f"got {ttir.count('scf.for')}\n{ttir}")

        err_g = (c_g.float() - ref_g).abs().max().item()
        err_fc = (c_fc.float() - ref_fc).abs().max().item()
        err_carry = (c_carry - ref_carry).abs().max().item()
        assert err_g < 2e-1, f"loop_distribute={loop_distribute}: acc_g max abs error {err_g} exceeds 2e-1"
        assert err_fc < 2e-1, f"loop_distribute={loop_distribute}: acc_fc max abs error {err_fc} exceeds 2e-1"
        assert err_carry < 5e-1, f"loop_distribute={loop_distribute}: carry max abs error {err_carry} exceeds 5e-1"

        results[loop_distribute] = (c_g.clone(), c_fc.clone(), c_carry.clone())

    # loop_distribute=False and loop_distribute=True must agree with each other.
    for i in range(3):
        torch.testing.assert_close(results[False][i].float(), results[True][i].float(), rtol=1e-2, atol=1e-2)


def _run_dual_dot_ptr_kernel(device, block, **options):
    """Compile and run `_dual_dot_ptr_kernel` on `block`x`block` tiles, check the
    result against a torch reference and return the compiled TTIR."""
    M = N = block
    K, BLOCK_K = 256, 64

    torch.manual_seed(17)
    a = torch.randn((M, K), dtype=torch.float16, device=device)
    w_g = torch.randn((K, N), dtype=torch.float16, device=device)
    w_fc = torch.randn((K, N), dtype=torch.float16, device=device)
    c_g = torch.empty((M, N), dtype=torch.float16, device=device)
    c_fc = torch.empty((M, N), dtype=torch.float16, device=device)

    kernel = _dual_dot_ptr_kernel[(1, 1)](
        a,
        w_g,
        w_fc,
        c_g,
        c_fc,
        K,
        a.stride(0),
        a.stride(1),
        w_g.stride(0),
        w_g.stride(1),
        w_fc.stride(0),
        w_fc.stride(1),
        c_g.stride(0),
        c_g.stride(1),
        BLOCK_M=M,
        BLOCK_N=N,
        BLOCK_K=BLOCK_K,
        **options,
    )

    err_g = (c_g.float() - torch.matmul(a.float(), w_g.float())).abs().max().item()
    err_fc = (c_fc.float() - torch.matmul(a.float(), w_fc.float())).abs().max().item()
    assert err_g < 2e-1, f"{options}: acc_g max abs error {err_g} exceeds 2e-1"
    assert err_fc < 2e-1, f"{options}: acc_fc max abs error {err_fc} exceeds 2e-1"

    return kernel.asm["ttir"]


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
@pytest.mark.parametrize(
    "block, num_warps, grf_mode, expected_loops",
    [
        # Two BLOCK_M x BLOCK_N f32 accumulators, so the fused loop holds
        # 2 * block * block * 4 / num_warps bytes per thread; the budget is
        # 4096 bytes for grf_mode='128' and 8192 for '256' and 'default'.
        (128, 4, 'default', 2),  # 32768 > 8192: distribute
        (128, 16, 'default', 1),  # 8192, not greater: leave fused
        (64, 4, 'default', 1),  # 8192, not greater: leave fused
        (64, 4, '128', 2),  # 8192 > 4096: distribute
        (64, 4, '256', 1),  # 8192, not greater: leave fused
    ],
)
def test_dual_dot_ptr_gemm_loop_distribute_cost_model(device, block, num_warps, grf_mode, expected_loops):
    """The cost model distributes a loop only when its accumulators exceed the
    register budget, so the verdict must track both `num_warps` and
    `grf_mode`."""
    ttir = _run_dual_dot_ptr_kernel(device, block, num_warps=num_warps, grf_mode=grf_mode,
                                    loop_distribute_cost_model=True)
    assert ttir.count("scf.for") == expected_loops, (
        f"block={block} num_warps={num_warps} grf_mode={grf_mode}: expected {expected_loops} "
        f"scf.for in ttir, got {ttir.count('scf.for')}\n{ttir}")


@pytest.mark.skipif(not is_xpu(), reason="XPU-specific test")
def test_loop_distribute_force_beats_cost_model(device):
    """`loop_distribute` distributes every legal loop, so it must win over the
    cost model on a loop the cost model rejects (64x64 f32 accumulators at 4
    warps sit exactly at the default budget)."""
    gated = _run_dual_dot_ptr_kernel(device, 64, num_warps=4, loop_distribute_cost_model=True)
    assert gated.count("scf.for") == 1, f"cost model should have rejected this loop\n{gated}"

    forced = _run_dual_dot_ptr_kernel(device, 64, num_warps=4, loop_distribute=True, loop_distribute_cost_model=True)
    assert forced.count("scf.for") == 2, f"loop_distribute should have distributed this loop\n{forced}"
