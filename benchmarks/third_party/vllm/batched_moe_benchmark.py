# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Batched MoE benchmark
=====================

This benchmark is based on the test_batched_moe.py tests and follows
the framework from gemm_benchmark.py to compare performance of different
batched MoE implementations using vLLM kernels.

"""
import os
from typing import Optional, List

import torch
import triton
import triton.language as tl

import triton_kernels_benchmark as benchmark_suite

# Import vLLM MoE functions
from vllm.model_executor.layers.fused_moe.fused_batched_moe import invoke_moe_batched_triton_kernel
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.utils import normalize_batched_scales_shape

# Import utility functions from vLLM tests
from tests.kernels.moe.utils import make_quantized_test_activations, make_test_weights
from tests.kernels.quant_utils import native_batched_masked_quant_matmul


@triton.jit
def moe_mmk(
    a_desc,
    b_desc,
    K,
    expert_id,
    a_scale_ptr,
    b_scale_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # Offsets and masks
    offs_m,
    offs_n,
    offs_bn,
    mask_m,
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    pid_m,
    pid_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    compute_type: tl.constexpr,
    use_w8a8: tl.constexpr,
    use_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
):

    if use_w8a16:
        b_scale_ptrs = b_scale_ptr + expert_id * stride_bse + offs_n[None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

    if use_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = b_scale_ptr + offs_bsn * stride_bsn

        # per act token
        elif per_act_token_quant:
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + offs_m * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=mask_m, other=0.0)[:, None]

            b_scale_ptrs = b_scale_ptr + offs_bn[None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)

        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load the next block of A and B using tensor descriptors
        a = a_desc.load([pid_m * BLOCK_M, k * BLOCK_K])
        b = b_desc.load([k * BLOCK_K, pid_n * BLOCK_N])

        # We accumulate along the K dimension.
        if use_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=mask_m, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # acc used to enable fp8_fast_accum
                accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)

    if use_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
    elif use_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    return accumulator


@triton.jit
def expert_triton_kernel(
    a_desc,  #[max_tokens, K]
    b_desc,  #[K, N]
    c_desc,  #[max_tokens, N]
    expert_id,
    compute_type: tl.constexpr,
    # Dimensions
    M,
    N,
    K,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    # strides
    stride_ak: tl.int64,
    stride_bk: tl.int64,
    stride_ase: tl.int64,
    stride_asm: tl.int64,
    stride_ask: tl.int64,
    stride_bse: tl.int64,
    stride_bsk: tl.int64,
    stride_bsn: tl.int64,
    # offsets
    offs_bn,
    # Blockwise quantization data
    group_n,
    group_k,
    pid_m,
    pid_n,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N) % N
    # offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    accumulator = moe_mmk(
        a_desc, b_desc, K, expert_id, a_scale_ptr, b_scale_ptr,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn,
        # Offsets and masks
        offs_m, offs_n, offs_bn, mask_m,
        # Block size for block-wise quantization
        group_n, group_k, pid_m, pid_n,
        # Meta-parameters
        BLOCK_M, BLOCK_N, BLOCK_K, compute_type, use_fp8_w8a8, use_int8_w8a16, per_act_token_quant)

    # store in C
    # offs_cn = tl.arange(0, BLOCK_N)
    # c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    # c_mask = mask_m[:, None] & (offs_cn[None, :] < N)
    c_desc.store([pid_m * BLOCK_M, pid_n * BLOCK_N], accumulator)
    # tl.store(c_ptrs, accumulator, mask=c_mask)


def get_matmul_batched_autotune_configs() -> List[triton.Config]:
    configs = [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'grf_mode': 'large'}, num_stages=s, num_warps=32)
        for s in [2, 3]
    ] + [
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'grf_mode': m}, num_stages=s, num_warps=w)
        for s in [2]
        for (m, w) in ([('large', 32), ('small', 64)])
    ] + [
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'grf_mode': 'large'}, num_stages=s, num_warps=32)
        for s in [2]
    ] + [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 512, 'BLOCK_K': 64, 'grf_mode': 'large'}, num_stages=s, num_warps=32)
        for s in [2]
    ] + [
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64, 'grf_mode': 'large'}, num_stages=s, num_warps=4)
        for s in [2]
    ]
    return configs


@triton.autotune(
    configs=get_matmul_batched_autotune_configs(),
    key=['max_num_tokens', 'K', 'N'],
)
@triton.jit
def batched_triton_kernel(
    a_ptr,  # [E, max_num_tokens, K]
    b_ptr,  # [E, K, N]
    c_ptr,  # [E, max_num_tokens, N]
    expert_num_tokens,  # [E]
    compute_type: tl.constexpr,
    # Dimensions
    max_num_tokens: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    # Quantization data
    a_scale_ptr,
    b_scale_ptr,
    b_zp_ptr,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_ae: tl.int64,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.int64,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ce: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_ase: tl.constexpr,
    stride_asm: tl.constexpr,
    stride_ask: tl.constexpr,
    stride_bse: tl.constexpr,
    stride_bsk: tl.constexpr,
    stride_bsn: tl.constexpr,
    # Blockwise quantization data
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Quantization schemes
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_act_token_quant: tl.constexpr,
    # Kernel config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    expert_id = tl.program_id(axis=0)
    e_num_tokens = tl.load(expert_num_tokens + expert_id)
    if e_num_tokens == 0:
        # Early exit
        return

    # axis 1 is M_blocks * N_blocks
    pid_mn = tl.program_id(axis=1)
    #num_pid_m = tl.cdiv(max_num_tokens, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    cta_m_start = pid_m * BLOCK_M
    cta_n_start = pid_n * BLOCK_N
    if cta_m_start >= e_num_tokens:
        # Early exit
        return

    cta_m_size = min(BLOCK_M, e_num_tokens - cta_m_start)
    cta_n_size = min(BLOCK_N, N - cta_n_start)

    a_desc = tl.make_tensor_descriptor(base=a_ptr + expert_id * stride_ae, shape=(e_num_tokens, K),
                                       strides=(stride_am, stride_ak), block_shape=(BLOCK_M, BLOCK_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr + expert_id * stride_be, shape=(K, N), strides=(stride_bk, stride_bn),
                                       block_shape=(BLOCK_K, BLOCK_N))
    c_desc = tl.make_tensor_descriptor(base=c_ptr + expert_id * stride_ce, shape=(e_num_tokens, N),
                                       strides=(stride_cm, stride_cn), block_shape=(BLOCK_M, BLOCK_N))

    # a_ptr = a_ptr + expert_id * stride_ae + cta_m_start * stride_am
    # b_ptr = b_ptr + expert_id * stride_be + cta_n_start * stride_bn
    # c_ptr = (c_ptr + expert_id * stride_ce + cta_m_start * stride_cm +
    #          cta_n_start * stride_cn)

    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)) % N

    if use_fp8_w8a8:
        a_scale_ptr = a_scale_ptr + expert_id * stride_ase
        b_scale_ptr = b_scale_ptr + expert_id * stride_bse

        # block-wise
        if group_k > 0 and group_n > 0 or per_act_token_quant:
            a_scale_ptr = a_scale_ptr + cta_m_start * stride_asm

    expert_triton_kernel(a_desc, b_desc, c_desc, expert_id, compute_type, cta_m_size,  # M
                         cta_n_size,  # N
                         K,  # K
                         a_scale_ptr, b_scale_ptr,
                         # Strides
                         stride_ak, stride_bk, stride_ase, stride_asm, stride_ask, stride_bse, stride_bsk, stride_bsn,
                         # offsets
                         offs_bn,
                         # Blockwise quantization data
                         group_n, group_k, pid_m, pid_n,
                         # Quantization schemes
                         use_fp8_w8a8, use_int8_w8a16, per_act_token_quant,
                         # Kernel config
                         BLOCK_M, BLOCK_N, BLOCK_K)


def invoke_moe_batched_triton_kernel_td(A: torch.Tensor,  # [E, max_tokens, K]
                                        B: torch.Tensor,  # [E, N, K]
                                        C: torch.Tensor,  # [E, max_tokens, N]
                                        expert_num_tokens: torch.Tensor,  # [E]
                                        compute_type: tl.dtype,
                                        # Quantization data
                                        A_scale: Optional[torch.Tensor], B_scale: Optional[torch.Tensor],
                                        B_zp: torch.Tensor,
                                        # Quantization schemes
                                        use_fp8_w8a8: bool, use_int8_w8a16: bool, use_int4_w4a16: bool,
                                        config: dict[str, int], per_act_token_quant: bool,
                                        block_shape: Optional[list[int]] = None):
    assert not use_int4_w4a16
    max_num_tokens = A.size(1)
    K = A.size(2)
    N = C.size(2)

    BLOCK_M = config['BLOCK_SIZE_M']
    BLOCK_N = config['BLOCK_SIZE_N']
    # BLOCK_K = config['BLOCK_SIZE_K']
    # Looks like generally good parameters
    # BLOCK_M = 256
    # BLOCK_N = 128
    # BLOCK_K = 32
    # num_warps = 64

    grid = (expert_num_tokens.size(0), triton.cdiv(max_num_tokens, BLOCK_M) * triton.cdiv(B.size(1), BLOCK_N))

    A_scale = normalize_batched_scales_shape(A_scale, expert_num_tokens.shape[0])

    if B_scale is not None and B_scale.ndim == 1:
        assert B_scale.numel() == expert_num_tokens.shape[0]
        B_scale = B_scale.view(-1, 1, 1)

    assert A_scale is None or A_scale.ndim == 3, (f'{0 if A_scale is None else A_scale.shape}')
    assert B_scale is None or B_scale.ndim == 1 or B_scale.ndim == 3, (f'{0 if B_scale is None else B_scale.shape}')

    if B_scale is not None:
        if B_scale.ndim == 1:
            stride_bse = 1
            stride_bsk = 0
            stride_bsn = 0
        else:
            stride_bse = B_scale.stride(0)
            stride_bsk = B_scale.stride(2)
            stride_bsn = B_scale.stride(1)

    else:
        stride_bse = 0
        stride_bsk = 0
        stride_bsn = 0

    if A_scale is not None:
        stride_ase = A_scale.stride(0)
        stride_asm = A_scale.stride(1)
        stride_ask = A_scale.stride(2)
    else:
        stride_ase = 0
        stride_asm = 0
        stride_ask = 0

    batched_triton_kernel[grid](
        A,
        B,
        C,
        expert_num_tokens,
        compute_type,
        # Dimensions
        max_num_tokens,
        K,
        N,
        # Quantization data
        A_scale,
        B_scale,
        B_zp,
        # Strides
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        stride_ase,
        stride_asm,
        stride_ask,
        stride_bse,
        stride_bsk,
        stride_bsn,
        # Blockwise quantization data
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        # Quantization schemes
        use_fp8_w8a8,
        use_int8_w8a16,
        per_act_token_quant,
        # Kernel config
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # BLOCK_K=BLOCK_K,
        # num_warps=num_warps,
    )


# Benchmark shapes for batched MoE
# (E: num_experts, M: max_tokens_per_expert, K: hidden_dim, N: intermediate_dim, fp8, block_quant)
# BATCHED_MM_X_VALS = [(E, M, K, N, False, False) for E in [8, 32] for M in [32, 224, 512] for K in [128, 1024] for N in [128, 1024]]
# Each pair represent transformation for SwiGLU and then output transformation
MM_CONFIGS_BF16 = sum([[(E, M, hidden_size, int_size * 2, fp8, block_quant),  # input -> swiglu input
                        (E, M, int_size, hidden_size, fp8, block_quant)]  # swiglu output -> final output
                       for M in [1, 8, 64, 256]
                       for E, hidden_size, int_size, fp8, block_quant in [
                           # llama4 scout bf16
                           (16, 5120, 8192, False, False),
                           # gpt-oss 20b mxfp4
                           (32, 2880, 2880, False, False),
                           # gpt-oss 120b mxfp4
                           (128, 2880, 2880, False, False),
                           # qwen3-235b-A22B bf16/fp8
                           (128, 4096, 1536, False, False),
                           # qwen3-30b-A3B bf16/fp8
                           (128, 2048, 768, False, False),
                           # qwen3-next-80B bf16
                           (512, 2048, 512, False, False),
                       ]], [])

MM_CONFIGS_FP8 = sum([[(E, M, hidden_size, int_size * 2, fp8, block_quant),
                       (E, M, int_size, hidden_size, fp8, block_quant)]
                      for M in [1, 8, 64, 256]
                      for E, hidden_size, int_size, fp8, block_quant in [
                          # deepseek V3, fp8 block quant
                          # 3.5 GBs of weights!
                          (256, 7168, 2048, True, True),
                          #  # qwen3-235b-A22B bf16/fp8
                          (128, 4096, 1536, True, False),
                          # qwen3-30b-A3B bf16/fp8
                          (128, 2048, 768, True, False),
                      ]], [])

DEVICE_NAME = torch.xpu.get_device_name()
DEVICE_TOTAL_MEMORY = torch.xpu.get_device_properties().total_memory


def is_enough_memory(x_val):
    E, M, K, N, fp8, block_quant = x_val
    # A: (E, M, K) bfloat16 or fp8
    # B: (E, K, N) bfloat16 or fp8
    # C: (E, M, N) bfloat16
    # num_expert_tokens: (E,) int32
    n_bytes = 1 if fp8 else 2
    required_memory = E * M * K * n_bytes + E * K * N * n_bytes + E * M * N * 2 + E * 4
    enough_memory = required_memory < DEVICE_TOTAL_MEMORY
    if not enough_memory:
        print(f"'{x_val}' combination skipped for '{DEVICE_NAME}'; {required_memory=} but {DEVICE_TOTAL_MEMORY=}")
    return enough_memory


MM_CONFIGS_BF16 = [x_val for x_val in MM_CONFIGS_BF16 if is_enough_memory(x_val)]


def get_batched_mm_benchmark(
    providers_filter: Optional[list[str]] = None,
    per_act_token_quant: bool = False,
    fp8=False,
    plot_name: str = 'moe-gemm-performance',
):
    """
    Returns a Mark object containing a Benchmark object for batched matrix multiplication.
    """
    supported_providers = {
        'triton': 'triton',
        'triton-td': 'triton-td',
        'pytorch': 'pytorch',
    }
    if fp8:
        # pytorch is very slow with fp8 case, for (8, 64, 1024, 2048) case it has ~0.15 TFlops vs 1.5 for triton
        del supported_providers['pytorch']

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = MM_CONFIGS_FP8 if fp8 else MM_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['num_experts', 'max_tokens_per_expert', 'K', 'N', 'fp8', 'block_quant'],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name=plot_name,
            args={},
        ))
    def benchmark(num_experts, max_tokens_per_expert, K, N, fp8, block_quant, provider):
        current_platform.seed_everything(70)
        n_warmup = 600

        act_dtype = torch.bfloat16
        use_fp8_w8a8 = fp8
        quant_dtype = torch.float8_e4m3fn if fp8 else None

        block_shape = (128, 128) if block_quant else None

        # Create random number of expert tokens
        num_expert_tokens = torch.randint(low=0, high=max_tokens_per_expert + 1, size=(num_experts, ), device='xpu',
                                          dtype=torch.int32)
        out_shape = (num_experts, max_tokens_per_expert, N)

        # Create quantized test activations
        A, A_q, A_scale = make_quantized_test_activations(
            num_experts,
            max_tokens_per_expert,
            K,
            in_dtype=act_dtype,
            quant_dtype=quant_dtype,
            block_shape=block_shape,
            per_act_token_quant=per_act_token_quant,
        )

        # Create test weights (only need B matrix for batched MM)
        (B, B_q, B_scale, _), _ = make_test_weights(
            num_experts,
            N // 2,
            K,
            in_dtype=act_dtype,
            quant_dtype=quant_dtype,
            block_shape=block_shape,
            per_act_token_quant=per_act_token_quant,
        )
        quantiles = [0.5, 0.0, 1.0]

        C = torch.zeros(out_shape, device='xpu', dtype=act_dtype)
        compute_tl_dtype = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}[C.dtype]
        rtol = 6e-2 if act_dtype == torch.bfloat16 else 1e-2
        atol = 6e-2 if act_dtype == torch.bfloat16 else 1e-2
        ref = torch.zeros(out_shape, device='xpu', dtype=act_dtype)

        def torch_fn():
            native_batched_masked_quant_matmul(A_q, B_q, ref, num_expert_tokens, A_scale, B_scale, block_shape,
                                               per_act_token_quant)
            return ref

        if provider == 'pytorch':
            # PyTorch reference implementation using native_batched_masked_quant_matmul
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                torch_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'triton' or provider == 'triton-td':
            # Triton batched MoE kernel
            invoke_kernel = invoke_moe_batched_triton_kernel if provider == 'triton' else invoke_moe_batched_triton_kernel_td

            def triton_fn():
                invoke_kernel(
                    A_q,
                    B_q,
                    C,
                    num_expert_tokens,
                    compute_tl_dtype,
                    A_scale,
                    B_scale,
                    None,
                    use_fp8_w8a8,
                    False,
                    False,
                    config={'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32 if fp8 else 16},
                    # config={'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32 if dtype.itemsize > 1 else 32},
                    per_act_token_quant=per_act_token_quant,
                    block_shape=block_shape,
                )
                return C

            # Verify correctness against reference
            benchmark_suite.assert_close(triton_fn, torch_fn, atol=atol, rtol=rtol, err_msg='triton to torch')
            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        # Calculate performance metrics
        # Memory bandwidth: A (E*M*K*2) + B (E*K*N*2) + C (E*M*N*4) bytes
        # Compute: E * M * N * K * 2 FLOPs (multiply-add)

        def gbps(ms):
            n_bytes = 1 if fp8 else 2
            total_bytes = num_experts * (max_tokens_per_expert * K * n_bytes + K * N * n_bytes +
                                         max_tokens_per_expert * N * 2)
            return total_bytes * (1e-9) / (ms * 1e-3)

        def tflops(ms):
            total_flops = num_experts * max_tokens_per_expert * N * K * 2
            return total_flops * (1e-12) / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


if __name__ == '__main__':
    # Run batched MM benchmark
    print('Running batched MM benchmark...')
    _benchmark_mm = get_batched_mm_benchmark(fp8=(os.getenv('FP8', '0') == '1'), )
    _benchmark_mm.run(show_plots=False, print_data=True)
