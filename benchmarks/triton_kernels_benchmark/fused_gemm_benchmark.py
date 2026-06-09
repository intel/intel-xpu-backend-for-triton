import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from torch.profiler import profile, ProfilerActivity


def native_torch_gemms(x, w_g, w_fc, b_g, b_fc):
    gate = torch.nn.functional.silu(torch.nn.functional.linear(x, w_g.T, b_g))
    fc = torch.nn.functional.linear(x, w_fc.T, b_fc)
    y = gate * fc
    return y


def fwd_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3,
                      num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3,
                      num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3,
                      num_warps=16),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=3,
                      num_warps=32),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 4}, num_stages=3,
                      num_warps=16),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 4}, num_stages=3,
                      num_warps=16),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3,
                      num_warps=32),
    ]
    return configs


@triton.autotune(configs=fwd_autotune_config(), key=["N", "K", "IS_TRAINING"])
@triton.jit
def fused_gemm_post_op_kernel(
    x_ptr,
    w_g_ptr,
    w_fc_ptr,
    b_g_ptr,
    b_fc_ptr,
    y_ptr,
    M,
    N,
    K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    dtype = y_ptr.type.element_ty
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if (pid_m * BLOCK_SIZE_M >= M) or (pid_n * BLOCK_SIZE_N >= N):
        return

    desc_x = tl.make_tensor_descriptor(
        x_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )

    desc_wg = tl.make_tensor_descriptor(
        w_g_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )

    desc_wfc = tl.make_tensor_descriptor(
        w_fc_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
    )

    offset_wn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    b_g = tl.load(b_g_ptr + offset_wn, mask=offset_wn < N, other=0.0)
    b_fc = tl.load(b_fc_ptr + offset_wn, mask=offset_wn < N, other=0.0)

    off_m = pid_m * BLOCK_SIZE_M
    off_n = pid_n * BLOCK_SIZE_N

    acc_g = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc_fc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, K, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        x = desc_x.load([off_m, k])
        w_g = desc_wg.load([k, off_n])
        w_fc = desc_wfc.load([k, off_n])

        acc_g = tl.dot(x, w_g, acc_g)
        acc_fc = tl.dot(x, w_fc, acc_fc)

    acc_g += b_g[None, :]
    acc_fc += b_fc[None, :]
    acc_g = acc_g.to(dtype)
    acc_fc = acc_fc.to(dtype)

    dtype = acc_g.type.element_ty
    acc_g = acc_g.to(tl.float32)
    silu_g = libdevice.fast_dividef(acc_g, 1.0 + libdevice.fast_expf(-acc_g)).to(dtype)
    acc_product = silu_g.to(tl.float32) * acc_fc.to(tl.float32)
    y = acc_product.to(dtype)

    desc_y = tl.make_tensor_descriptor(
        y_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    desc_y.store([off_m, off_n], y)


if __name__ == "__main__":
    dtype = torch.bfloat16
    x = torch.randn((220000, 512), dtype=dtype, device="xpu").requires_grad_()
    w_g = torch.randn((512, 1024), dtype=dtype, device="xpu").requires_grad_()
    w_fc = torch.randn((512, 1024), dtype=dtype, device="xpu").requires_grad_()
    b_g = torch.randn((1024), dtype=dtype, device="xpu").requires_grad_()
    b_fc = torch.randn((1024), dtype=dtype, device="xpu").requires_grad_()
    assert w_g.shape == w_fc.shape
    assert b_g.shape == b_fc.shape
    assert x.shape[1] == w_g.shape[0]
    assert b_g.shape[0] == w_g.shape[1]

    total_len, in_dim = x.shape
    out_dim = w_g.shape[1]
    y = x.new_empty(total_len, out_dim)
    grid = lambda META: (triton.cdiv(total_len, META['BLOCK_SIZE_M']) * triton.cdiv(out_dim, META['BLOCK_SIZE_N']), )
    fused_gemm_post_op_kernel[grid](
        x,
        w_g,
        w_fc,
        b_g,
        b_fc,
        y,
        total_len,
        out_dim,
        in_dim,
    )

    ref = native_torch_gemms(x, w_g, w_fc, b_g, b_fc)
    atol = 1e-2
    rtol = 1e-2
    if torch.allclose(y, ref, atol=atol, rtol=rtol):
        print("pass")
    else:
        print("fail")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
        for _ in range(100):
            fused_gemm_post_op_kernel[grid](
                x,
                w_g,
                w_fc,
                b_g,
                b_fc,
                y,
                total_len,
                out_dim,
                in_dim,
            )
    print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=-1))
