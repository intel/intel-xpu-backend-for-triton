# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
"""
Grouped GEMM benchmark (vLLM XPU op vs Triton)
==============================================
"""
from typing import Optional

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite


def _has_cutlass_grouped_gemm_interface() -> bool:
    return hasattr(getattr(torch.ops, "_xpu_C", None), "cutlass_grouped_gemm_interface")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "grf_mode": "256"},
                      num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "grf_mode": "256"},
                      num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "grf_mode": "256"},
                      num_warps=32, num_stages=2),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "grf_mode": "256"},
                      num_warps=32, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def grouped_gemm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
    d_ptr,
    rows_ptr,
    offsets_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_biase: tl.constexpr,
    stride_biasn: tl.constexpr,
    stride_dm: tl.constexpr,
    stride_dn: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    expert_id = tl.program_id(axis=1)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    m_expert = tl.load(rows_ptr + expert_id).to(tl.int64)
    if m_expert == 0:
        return

    num_pid_m = tl.cdiv(m_expert, BLOCK_SIZE_M)
    if pid_m >= num_pid_m:
        return

    expert_offset = tl.load(offsets_ptr + expert_id).to(tl.int64)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < m_expert
    mask_n = offs_n < N

    global_rows = expert_offset + offs_m

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < K

        a_ptrs = a_ptr + global_rows[:, None] * stride_am + k_idx[None, :] * stride_ak
        b_ptrs = b_ptr + expert_id.to(tl.int64) * stride_be + k_idx[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    if HAS_BIAS:
        bias_ptrs = bias_ptr + expert_id.to(tl.int64) * stride_biase + offs_n * stride_biasn
        bias = tl.load(bias_ptrs, mask=mask_n, other=0.0).to(tl.float32)
        acc += bias[None, :]

    d_ptrs = d_ptr + global_rows[:, None] * stride_dm + offs_n[None, :] * stride_dn
    tl.store(d_ptrs, acc.to(tl.bfloat16), mask=mask_m[:, None] & mask_n[None, :])


def grouped_gemm_triton(
    ptr_A: torch.Tensor,
    ptr_B: torch.Tensor,
    ptr_D: torch.Tensor,
    rows_per_expert: torch.Tensor,
    ptr_bias: Optional[torch.Tensor] = None,
):
    if ptr_A.dtype != torch.bfloat16 or ptr_B.dtype != torch.bfloat16 or ptr_D.dtype != torch.bfloat16:
        raise AssertionError("grouped_gemm_triton currently supports bfloat16 tensors only")
    total_m = int(rows_per_expert.sum().item())
    if total_m == 0:
        return ptr_D
    if ptr_A.shape[0] != total_m:
        raise AssertionError(f"Expected ptr_A.shape[0] == sum(rows_per_expert), got {ptr_A.shape[0]} vs {total_m}")
    if ptr_B.ndim != 3:
        raise AssertionError("ptr_B must have shape [num_experts, K, N]")
    if ptr_D.shape[0] != total_m or ptr_D.shape[1] != ptr_B.shape[2]:
        raise AssertionError("ptr_D must have shape [sum(rows_per_expert), N]")

    offsets = torch.cumsum(rows_per_expert, dim=0, dtype=torch.int64) - rows_per_expert.to(torch.int64)

    max_rows_per_expert = int(rows_per_expert.max().item())

    def compute_grid(META):
        return (
            triton.cdiv(max_rows_per_expert, META["BLOCK_SIZE_M"]) * triton.cdiv(ptr_B.shape[2], META["BLOCK_SIZE_N"]),
            ptr_B.shape[0],
        )

    dummy_bias = ptr_D.new_zeros((ptr_B.shape[0], ptr_B.shape[2]))
    bias = ptr_bias if ptr_bias is not None else dummy_bias

    grouped_gemm_kernel[compute_grid](
        ptr_A,
        ptr_B,
        bias,
        ptr_D,
        rows_per_expert,
        offsets,
        N=ptr_B.shape[2],
        K=ptr_B.shape[1],
        stride_am=ptr_A.stride(0),
        stride_ak=ptr_A.stride(1),
        stride_be=ptr_B.stride(0),
        stride_bk=ptr_B.stride(1),
        stride_bn=ptr_B.stride(2),
        stride_biase=bias.stride(0),
        stride_biasn=bias.stride(1),
        stride_dm=ptr_D.stride(0),
        stride_dn=ptr_D.stride(1),
        HAS_BIAS=ptr_bias is not None,
    )
    return ptr_D


MM_CONFIGS = [
    # (num_experts, max_tokens_per_expert, K, N, has_bias)
    (16, 128, 4096, 4096, False),
    (16, 128, 4096, 4096, True),
    (32, 128, 2048, 1536, False),
    (32, 128, 2048, 1536, True),
    (128, 64, 2048, 4096, False),
    (128, 64, 2048, 4096, True),
]


def get_grouped_gemm_benchmark(providers_filter: Optional[list[str]] = None):
    supported_providers = {"triton": "triton"}
    if _has_cutlass_grouped_gemm_interface():
        supported_providers["xpu-cutlass"] = "xpu-cutlass"
    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=["num_experts", "max_tokens_per_expert", "K", "N", "has_bias"],
            x_vals=MM_CONFIGS,
            line_arg="provider",
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[("green", "-"), ("blue", "--"), ("red", ":")],
            ylabel=["GB/s", "TFlops"],
            plot_name="grouped-gemm-performance",
            args={},
        ))
    def benchmark(num_experts, max_tokens_per_expert, K, N, has_bias, provider):
        torch.manual_seed(2025)
        rows_per_expert = torch.randint(
            low=0,
            high=max_tokens_per_expert + 1,
            size=(num_experts, ),
            dtype=torch.int32,
            device="xpu",
        )
        total_m = int(rows_per_expert.sum().item())
        if total_m == 0:
            rows_per_expert[0] = 1
            total_m = 1

        # expert_first_token_offset[i] is the first token row index for expert i
        # in the flattened activation matrix (exclusive prefix sum of rows_per_expert).
        expert_first_token_offset = (
            torch.cumsum(rows_per_expert.to(torch.int64), dim=0) - rows_per_expert.to(torch.int64)
        )

        ptr_A = torch.randn((total_m, K), device="xpu", dtype=torch.bfloat16)
        ptr_B = torch.randn((num_experts, K, N), device="xpu", dtype=torch.bfloat16)
        ptr_bias = torch.randn((num_experts, N), device="xpu", dtype=torch.bfloat16) if has_bias else None

        out_triton = torch.empty((total_m, N), device="xpu", dtype=torch.bfloat16)

        def triton_fn():
            grouped_gemm_triton(ptr_A=ptr_A, ptr_B=ptr_B, ptr_D=out_triton, rows_per_expert=rows_per_expert, ptr_bias=ptr_bias)
            return out_triton

        if provider == "triton":
            if _has_cutlass_grouped_gemm_interface():
                out_ref = torch.empty_like(out_triton)

                def ref_fn():
                    torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                        ptr_A=ptr_A,
                        ptr_B=ptr_B,
                        ptr_scales=None,
                        ptr_bias=ptr_bias,
                        ptr_D=out_ref,
                        expert_first_token_offset=expert_first_token_offset,
                        N=N,
                        K=K,
                        num_experts=num_experts,
                        is_B_int4=False,
                        is_B_mxfp4=False,
                    )
                    return out_ref

                benchmark_suite.assert_close(triton_fn, ref_fn, atol=1e-2, rtol=1e-2, err_msg="triton to xpu-cutlass")

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=200,
                n_repeat=10,
                quantiles=[0.5, 0.0, 1.0],
            )
        elif provider == "xpu-cutlass":
            out_cutlass = torch.empty_like(out_triton)

            def cutlass_fn():
                torch.ops._xpu_C.cutlass_grouped_gemm_interface(
                    ptr_A=ptr_A,
                    ptr_B=ptr_B,
                    ptr_scales=None,
                    ptr_bias=ptr_bias,
                    ptr_D=out_cutlass,
                    expert_first_token_offset=expert_first_token_offset,
                    N=N,
                    K=K,
                    num_experts=num_experts,
                    is_B_int4=False,
                    is_B_mxfp4=False,
                )
                return out_cutlass

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                cutlass_fn,
                n_warmup=200,
                n_repeat=10,
                quantiles=[0.5, 0.0, 1.0],
            )
        else:
            raise NotImplementedError(f"Unsupported provider {provider}")

        active_experts = int((rows_per_expert > 0).sum().item())
        input_bytes = ptr_A.element_size() * total_m * K
        weight_bytes = ptr_B.element_size() * active_experts * K * N
        output_bytes = out_triton.element_size() * total_m * N
        bias_bytes = (ptr_bias.element_size() * active_experts * N) if ptr_bias is not None else 0

        def gbps(ms):
            return (input_bytes + weight_bytes + output_bytes + bias_bytes) * 1e-9 / (ms * 1e-3)

        def tflops(ms):
            return (2 * total_m * K * N) * 1e-12 / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == "__main__":
    _benchmark = get_grouped_gemm_benchmark()
    _benchmark.run(show_plots=False, print_data=True)
