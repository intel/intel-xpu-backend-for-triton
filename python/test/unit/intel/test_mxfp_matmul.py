import pytest
import torch
import triton
import triton.language as tl
from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor


def f8_to_f16(x, dtype):

    @triton.jit
    def kernel(Y, X, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X + offs, mask=mask)
        tl.store(Y + offs, x, mask=mask)

    ret = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(x.numel(), META['BLOCK_SIZE']), )
    dtype = getattr(tl, dtype)
    kernel[grid](ret, triton.reinterpret(x, dtype), ret.numel(), BLOCK_SIZE=1024)
    return ret


def fp8e8m0_to_float32(scale):
    scale = scale.view(torch.uint8)
    scale = scale.to(torch.int32)
    scale = scale << 23
    scale = scale.view(torch.float32)
    return scale


@triton.jit
def mxfp_matmul(  #
        a_ptr, b_ptr, output_ptr,  #
        a_scale, b_scale,  #
        M, N, K,  #
        stride_scale,  #
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,  #
        DTYPE_A: tl.constexpr,  #
        DTYPE_B: tl.constexpr,  #
        BLOCK_M: tl.constexpr,  #
        BLOCK_N: tl.constexpr,  #
        BLOCK_K: tl.constexpr,  #
        NUM_STAGES: tl.constexpr,  #
        PACK_B_ALONG_K: tl.constexpr = True):
    DIV_FACTOR_A: tl.constexpr = 2 if DTYPE_A == "e2m1" else 1
    DIV_FACTOR_B: tl.constexpr = 2 if DTYPE_B == "e2m1" else 1
    DIV_FACTOR_B_K: tl.constexpr = DIV_FACTOR_B if PACK_B_ALONG_K else 1
    DIV_FACTOR_B_N: tl.constexpr = 1 if PACK_B_ALONG_K else DIV_FACTOR_B
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    offs_bn = (pid_n * BLOCK_N // DIV_FACTOR_B_N + tl.arange(0, BLOCK_N // DIV_FACTOR_B_N))
    offs_bn_scale = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_ak = tl.arange(0, BLOCK_K // DIV_FACTOR_A)
    offs_bk = tl.arange(0, BLOCK_K // DIV_FACTOR_B_K)
    offs_scale_k = tl.arange(0, BLOCK_K // 32)

    if a_scale is not None:
        a_scale_ptr = a_scale + offs_am[:, None] * stride_scale + offs_scale_k[None, :]
    if b_scale is not None:
        b_scale_ptr = b_scale + offs_bn_scale[:, None] * stride_scale + offs_scale_k[None, :]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=output_ptr.dtype.element_ty)

    for k in tl.range(0, tl.cdiv(K, BLOCK_K), num_stages=NUM_STAGES):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        if a_scale is not None:
            scale_a = tl.load(a_scale_ptr)
        else:
            scale_a = None
        if b_scale is not None:
            scale_b = tl.load(b_scale_ptr)
        else:
            scale_b = None
        accumulator = tl.dot_scaled(a, scale_a, DTYPE_A, b, scale_b, DTYPE_B, accumulator, rhs_k_pack=PACK_B_ALONG_K)
        a_ptrs += (BLOCK_K // DIV_FACTOR_A) * stride_ak
        b_ptrs += (BLOCK_K // DIV_FACTOR_B_K) * stride_bk
        if a_scale is not None:
            a_scale_ptr += BLOCK_K // 32
        if b_scale is not None:
            b_scale_ptr += BLOCK_K // 32

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_ptrs = output_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=c_mask)


@pytest.mark.parametrize("M, N, K", [(1024, 512, 512)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 128, 128)])
@pytest.mark.parametrize("NUM_STAGES", [1, 3])
@pytest.mark.parametrize("B_TRANS", [True, False])
@pytest.mark.parametrize("PACK_B_ALONG_K", [True, False])
@pytest.mark.parametrize("A_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("B_DATA_TYPE", ["float8e5", "float8e4nv", "float4"])
@pytest.mark.parametrize("WITH_A_SCALE", [True, False])
@pytest.mark.parametrize("WITH_B_SCALE", [True, False])
def test_mxfp_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES, B_TRANS, PACK_B_ALONG_K, A_DATA_TYPE, B_DATA_TYPE,
                     WITH_A_SCALE, WITH_B_SCALE, device):
    if not PACK_B_ALONG_K and B_DATA_TYPE != "float4":
        pytest.xfail("Pack along K can only be False for float4")

    if BLOCK_N == 256 and BLOCK_K == 256:
        NUM_STAGES = 2

    torch.manual_seed(42)

    def create_operand(dtype: str, size0: int, size1: int, k_dim: int, transpose: bool = True,
                       pack_along_k: bool = True):
        if dtype == "float8e5":
            if transpose:
                v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e5m2).to(device)
                v_ref = f8_to_f16(v.view(torch.float8_e5m2), dtype).to(torch.float32)
            else:
                v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e5m2).to(device).T
                v_ref = f8_to_f16(v.view(torch.float8_e5m2).T, dtype).to(torch.float32).T
        elif dtype == "float8e4nv":
            if transpose:
                v = torch.randint(20, 40, (size0, size1), dtype=torch.uint8).view(torch.float8_e4m3fn).to(device)
                v_ref = f8_to_f16(v.view(torch.float8_e4m3fn), dtype).to(torch.float32)
            else:
                v = torch.randint(20, 40, (size1, size0), dtype=torch.uint8).view(torch.float8_e4m3fn).to(device).T
                v_ref = f8_to_f16(v.view(torch.float8_e4m3fn).T, dtype).to(torch.float32).T
        else:
            # float4
            if pack_along_k:
                pack_dim = k_dim
            else:
                pack_dim = (k_dim + 1) % 2
            if transpose:
                v_mxfp4 = MXFP4Tensor(size=(size0, size1), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=pack_dim)
                v_ref = v_mxfp4.to(torch.float32)
            else:
                v_mxfp4 = MXFP4Tensor(size=(size1, size0), device=device).random()
                v = v_mxfp4.to_packed_tensor(dim=(pack_dim + 1) % 2).T
                v_ref = v_mxfp4.to(torch.float32).T
        return v, v_ref

    dtype_converter = {'float8e5': 'e5m2', 'float8e4nv': 'e4m3', 'float4': 'e2m1'}

    a, a_ref = create_operand(A_DATA_TYPE, M, K, 1)
    b, b_ref = create_operand(B_DATA_TYPE, K, N, 0, B_TRANS, PACK_B_ALONG_K)

    a_scale_mxfp4 = MXScaleTensor(size=(M, (K + 32 - 1) // 32), device=device).random(high=32.0)
    b_scale_mxfp4 = MXScaleTensor(size=(N, (K + 32 - 1) // 32), device=device).random(high=32.0)
    a_scale = a_scale_mxfp4.data
    b_scale = b_scale_mxfp4.data

    a_scale_ref = a_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1)[:M, :K]
    b_scale_ref = b_scale_mxfp4.to(torch.float32).repeat_interleave(32, dim=1).T.contiguous()[:K, :N]
    stride_scale = b_scale.stride(0)
    if not WITH_A_SCALE:
        a_scale = None
        a_scale_ref = 1.0
    if not WITH_B_SCALE:
        b_scale = None
        b_scale_ref = 1.0

    ref_out = torch.matmul(a_ref * a_scale_ref, b_ref * b_scale_ref)

    output = a.new_empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), 1)
    kernel_kwargs = {}
    mxfp_matmul[grid](a, b, output, a_scale, b_scale, M, N, K, stride_scale, a.stride(0), a.stride(1), b.stride(0),
                      b.stride(1), output.stride(0), output.stride(1), dtype_converter[A_DATA_TYPE],
                      dtype_converter[B_DATA_TYPE], BLOCK_M, BLOCK_N, BLOCK_K, PACK_B_ALONG_K=PACK_B_ALONG_K,
                      NUM_STAGES=NUM_STAGES, **kernel_kwargs)

    atol = 1e-3
    if WITH_A_SCALE and WITH_B_SCALE and A_DATA_TYPE == "float4" and B_DATA_TYPE == "float4" and not B_TRANS:
        # Looks like a common error in calculating real numbers.
        # Potential area for improvement.
        atol = 3e-3
    torch.testing.assert_close(ref_out, output, atol=atol, rtol=1e-3)
