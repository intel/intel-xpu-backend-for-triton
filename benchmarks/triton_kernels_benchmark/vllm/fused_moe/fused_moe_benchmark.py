# pylint: skip-file
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused MoE benchmark
===================

This benchmark measures the performance of the fused MoE grouped GEMM kernel
and follows the framework from gemm_benchmark.py to compare performance of
different implementations using vLLM kernels.

"""
import os
import random
from math import prod
from typing import Optional

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Any

import triton_kernels_benchmark as benchmark_suite

from tests.kernels.moe.utils import make_quantized_test_activations, make_test_weight
from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_triton_kernel, get_default_config
from vllm_xpu_kernels.fused_moe_interface import cutlass_grouped_gemm as sycl_tla_grouped_gemm

DEVICE = triton.runtime.driver.active.get_active_torch_device()

DEVICE_TOTAL_MEMORY_BYTES = benchmark_suite.get_total_gpu_memory_bytes()


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    # if envs.VLLM_BATCH_INVARIANT:
    #     return {
    #         "BLOCK_SIZE_M": 64,
    #         "BLOCK_SIZE_N": 64,
    #         "BLOCK_SIZE_K": 32,
    #         "GROUP_SIZE_M": 8,
    #         "SPLIT_K": 1,
    #     }

    # num_stages can cause triton.runtime.errors.OutOfResources on ROCm.
    num_stages_rocm = 2

    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: tile sizes are constrained by block_shape.
        # Use a small M tile for decode-like batches where tokens are
        # spread thin across experts. Larger batches benefit from
        # GROUP_SIZE_M > 1 because the per-block scales add memory
        # traffic that benefits from L2 tile reuse.
        # config = {
        #     "BLOCK_SIZE_M": 16 if M <= 64 else 64,
        #     "BLOCK_SIZE_N": block_shape[0],
        #     "BLOCK_SIZE_K": block_shape[1],
        #     "GROUP_SIZE_M": 1 if M <= 16 else 32,
        #     "SPLIT_K": 1,
        #     "num_warps": 4,
        #     "num_stages": 3 if not current_platform.is_rocm() else num_stages_rocm,
        # }
        pass
    elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
        # moe wna16 kernels
        # only set BLOCK_SIZE_M
        # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
        bit = 4 if dtype == "int4_w4a16" else 8
        use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk, block_shape[1], E, bit)
        if use_moe_wna16_cuda:
            config = {"BLOCK_SIZE_M": min(16, M), "SPLIT_K": 1}
        elif M <= 20:
            config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        elif M <= 40:
            config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
        else:
            config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1, "SPLIT_K": 1}
    else:
        # General defaults for bf16/fp16 and fp8 per-tensor.
        # Tile sizes scale with batch: small batches are memory-bound
        # (favor tall-K tiles), large batches are compute-bound (favor
        # large M/N tiles with more warps).
        if M <= 32:
            block_m = 16
        elif M <= 96:
            block_m = 32
        elif M <= 512:
            block_m = 64
        else:
            block_m = 128

        block_n = 64 if M <= 64 else 128

        # Small batches benefit from longer reduction (larger K tile),
        # while large batches prefer more output parallelism.
        # FP8 elements are half-width so larger K tiles are always cheap.
        block_k = 128 if dtype == "fp8_w8a8" or M <= 64 else 64

        # Grouping adjacent M-blocks lets them share weight tiles in L2.
        # Only helps when there are enough M-blocks per expert to group;
        # with many experts each one sees few tokens so grouping is useless.
        tokens_per_expert = M // max(E, 1)
        group_m = 16 if tokens_per_expert > 128 else 1

        # Large batches have enough blocks to saturate the GPU, so we
        # use more warps per block to increase arithmetic intensity.
        num_warps = 4 if M <= 128 else 8

        if False:  #current_platform.is_rocm():
            num_stages = num_stages_rocm
        elif M <= 32:
            num_stages = 4
        else:
            num_stages = 3

        config = {
            "BLOCK_SIZE_M": 256,  #block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": 64,  #block_k,
            "GROUP_SIZE_M": group_m,
            "SPLIT_K": 1,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def get_fused_moe_configs():
    return [
        triton.Config({'BLOCK_SIZE_N': N, 'GROUP_SIZE_M': G, 'SPLIT_K': 1, 'grf_mode': '256'}, num_stages=s,
                      num_warps=w)
        # for N in [32, 64, 128]
        # for G in [1, 8, 16]
        # for s in [3, 4]
        # for w in [4, 8, 16, 32]
        for N in [256]
        for G in [4]
        for s in [1]
        for w in [32]
    ]


@triton.autotune(
    configs=get_fused_moe_configs(),
    key=['M', 'top_k', 'K', 'N'],
)
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    - naive_block_assignment: A boolean flag indicating whether to use naive
        token wise block assignment. If True, each block corresponds to a
        single token.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(offs == 0, pid_m,  # first element = pid_m
                              num_valid_tokens,  # remaining elements = constant
                              )

    token_mask = offs_token < num_valid_tokens
    # offs_token = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    # if off_experts == -1:
    #     # -----------------------------------------------------------
    #     # Write back zeros to the output when the expert is not
    #     # in the current expert parallel rank.
    #     write_zeros_to_output(
    #         c_ptr,
    #         stride_cm,
    #         stride_cn,
    #         pid_n,
    #         N,
    #         offs_token,
    #         token_mask,
    #         BLOCK_SIZE_M,
    #         BLOCK_SIZE_N,
    #         compute_type,
    #     )
    #     return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                       block_shape=(1, BLOCK_SIZE_K))

    b_desc = tl.make_tensor_descriptor(base=b_ptr + off_experts * stride_be, shape=(N, K),
                                       strides=(stride_bn, stride_bk), block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K))
    if use_int8_w8a16:
        b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn)
        b_scale = tl.load(b_scale_ptrs)

    if use_fp8_w8a8 or use_int8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn)
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.gather(offs_token // top_k, k * BLOCK_SIZE_K)
        b = b_desc.load([pid_n * BLOCK_SIZE_N, k * BLOCK_SIZE_K]).T
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)

    # Dequantization for supported quantization schemes:
    #   - int8_w8a16
    #   - fp8_w8a8
    #   - int8_w8a8
    # Accumulator and scalings are in float32 to preserve numerical accuracy.
    if use_int8_w8a16:
        accumulator = accumulator * b_scale
    elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
        accumulator = accumulator * a_scale * b_scale

    # Bias addition:
    # Bias must be applied after dequantization:
    #   - Since bias is typically not quantized
    #   - Bias should not be scaled by quantization factors
    if HAS_BIAS:
        accumulator += bias[None, :]

    # Router (MoE) weight multiplication:
    # This multiplication MUST be performed in float32 before any precision
    # conversion to ensure numerical stability, which is especially critical
    # on ROCm platforms.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    # Final precision conversion:
    # Cast once at the end to the desired compute/output dtype.
    accumulator = accumulator.to(compute_type)

    # -----------------------------------------------------------
    # Write back the block of the output
    # offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    # tl.store(c_ptrs, accumulator, mask=c_mask)
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                       block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], accumulator)


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor | None,
    B_scale: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: list[int] | None = None,
    B_bias: torch.Tensor | None = None,
):
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(B.size(-2), block_shape[0]) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(B.size(-1), block_shape[1]) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            # optimize for small batch_size.
            # We assume that top_ids of each token is unique,
            # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
            # and we can skip some invalid blocks.
            EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]
    grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]), )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    # print("johnlu EM:", EM)
    # print("johnlu B.size(1):", B.size(1))
    fused_moe_kernel[grid](
        A, B, C,
        B_bias, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, A.size(0),
        B.size(1), B.size(2), EM, num_tokens, A.stride(0), A.stride(1), B.stride(0), B.stride(2), B.stride(1),
        C.stride(1), C.stride(2), A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0, B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0], 0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type, use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8, use_int8_w8a16=use_int8_w8a16, per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None), HAS_BIAS=HAS_BIAS, BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"]
        # **config,
    )


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def torch_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    pad_sorted_ids: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Golden torch implementation of moe_align_block_size.

    This function aligns the token distribution across experts to be compatible
    with block size for matrix multiplication by sorting tokens by expert and
    padding to block boundaries.
    """
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = topk_ids.numel() * block_size

    flattened_token_indices = torch.arange(topk_ids.numel(), device=topk_ids.device, dtype=torch.int32)
    flattened_expert_ids = topk_ids.flatten()
    sorted_expert_ids, sort_indices = torch.sort(flattened_expert_ids, stable=True)
    sorted_token_indices = flattened_token_indices[sort_indices]

    expert_token_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        mask = sorted_expert_ids == expert_id
        expert_token_counts[expert_id] = mask.sum()

    expert_padded_counts = torch.zeros(num_experts, dtype=torch.int64, device=topk_ids.device)
    for expert_id in range(num_experts):
        original_count = expert_token_counts[expert_id]
        if expert_map is not None and expert_map[expert_id] == -1:
            continue
        if original_count > 0:
            expert_padded_counts[expert_id] = ((original_count + block_size - 1) // block_size) * block_size

    sorted_token_ids = torch.full(
        (max_num_tokens_padded, ),
        topk_ids.numel(),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    max_num_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.zeros(max_num_blocks, dtype=torch.int32, device=topk_ids.device)

    current_pos = 0
    current_block = 0
    for expert_id in range(num_experts):
        if expert_map is not None and expert_map[expert_id] == -1:
            continue

        expert_mask = sorted_expert_ids == expert_id
        expert_tokens = sorted_token_indices[expert_mask]
        num_expert_tokens = expert_tokens.shape[0]

        if num_expert_tokens > 0:
            sorted_token_ids[current_pos:current_pos + num_expert_tokens] = expert_tokens

            expert_blocks_needed = expert_padded_counts[expert_id] // block_size
            expert_id_new = expert_id
            if expert_map is not None:
                expert_id_new = expert_map[expert_id]
            expert_ids[current_block:current_block + expert_blocks_needed] = expert_id_new

            current_pos += expert_padded_counts[expert_id]
            current_block += expert_blocks_needed

    total_padded_tokens = expert_padded_counts.sum()
    num_tokens_post_pad = torch.tensor([total_padded_tokens], dtype=torch.int32, device=topk_ids.device)

    return sorted_token_ids, expert_ids, num_tokens_post_pad


def is_enough_memory(x_val, safety_factor=0.80):
    M, N, K, num_experts, topk, dtype, has_bias = x_val

    fp8 = dtype == torch.float8_e4m3fn

    # A and B bf16 originals (always allocated, freed later in fp8 case)
    a_mem = M * K * 2
    b_mem = num_experts * K * N * 2

    if fp8:
        # fp8 copies + scales
        a_mem += M * K + 4
        b_mem += num_experts * K * N + num_experts * 4

    # C: output workspace (M, topk, N)
    out_mem = M * topk * N * 2

    required_memory = a_mem + b_mem + out_mem
    return required_memory < DEVICE_TOTAL_MEMORY_BYTES * safety_factor


def filter_by_memory(configs):
    result = []
    for x_val in configs:
        if is_enough_memory(x_val):
            result.append(x_val)
        else:
            print(f"'{x_val}' combination skipped, OOM expected")
    return result


# Benchmark configurations: (M, N, K, num_experts, topk, dtype, has_bias)
# Each tuple represents one fused MoE GEMM call with the given token count M.
MM_CONFIGS_BF16 = [
    # Qwen3-30B-A3B-Instruct w13 with MNK factors (80, 768 * 2 // 4, 2048), num_experts=128, topk=8
    # [80, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w2 with MNK factors (80, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8
    # [80, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w13 with MNK factors (8192, 768 * 2 // 4, 2048), num_experts=128, topk=8
    # [8192, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False],
    # Qwen3-30B-A3B-Instruct w2 with MNK factors (8192, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8
    # [8192, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False],
    # Llama-4-scout with MNK factors (30, 8192 * 2, 5120), num_experts=16, topk=1
    # [30, 8192 * 2, 5120, 16, 1, torch.bfloat16, False],
    # Llama-4-scout with MNK factors (8192, 8192 * 2, 5120), num_experts=16, topk=1
    # [8192, 8192 * 2, 5120, 16, 1, torch.bfloat16, False],
    [128, 128, 128, 1, 1, torch.bfloat16, False],
]

MM_CONFIGS_FP8 = [[M, N, K, num_experts, topk, torch.float8_e4m3fn, has_bias]
                  for M, N, K, num_experts, topk, _, has_bias in MM_CONFIGS_BF16]

MM_CONFIGS_BF16 = filter_by_memory(MM_CONFIGS_BF16)
MM_CONFIGS_FP8 = filter_by_memory(MM_CONFIGS_FP8)

# To debug if the benchmark runs at all, without waiting for all configurations to run
if os.getenv('DEBUG_BENCH', '0') == '1':
    MM_CONFIGS_BF16 = MM_CONFIGS_BF16[:1]
    MM_CONFIGS_FP8 = MM_CONFIGS_FP8[:1]


def _resize_cache(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor:
    """
    Shrink the given tensor and apply the given view to it.  This is
    used to resize the intermediate fused_moe caches.
    """
    assert prod(v) <= x.numel(), f"Requested view {v} with {prod(v)} elements exceeds tensor size {x.numel()}"
    return x.flatten()[:prod(v)].view(*v)


def _normalize_fp8_scale(scale: torch.Tensor | None) -> torch.Tensor | None:
    if scale is None:
        return None
    return scale.reshape(1) if scale.numel() == 1 else scale


def _dequantize_fp8(tensor: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    dequantized = tensor.to(torch.float32)
    if scale is not None:
        dequantized = dequantized * scale.to(torch.float32)
    return dequantized


RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}


def ref_prologue(
    x,
    scales,
    flat_expert_indices,
    num_per_tok,
    num_experts,
    recipe,
    ep_rank=0,
    ep_size=1,
):
    expert_start_id = num_experts * ep_rank
    expert_end_id = expert_start_id + num_experts

    idxs = flat_expert_indices.argsort(stable=True)
    counts = flat_expert_indices.bincount().cpu().numpy()
    tokens_per_expert = counts.cumsum()
    token_idxs = idxs // num_per_tok

    data_dtype, scale_dtype = RECIPE_TO_DTYPE.get(recipe, (None, None))
    if recipe in ["mxfp8", "mxfp4"]:
        scales = scales.view(torch.uint8)

    expand_input = []
    expand_scales = []
    for expert_id, end_idx in enumerate(tokens_per_expert):
        start_idx = 0 if expert_id == 0 else tokens_per_expert[expert_id - 1]
        if ((start_idx == end_idx) or (expert_id < expert_start_id) or (expert_id >= expert_end_id)):
            continue
        exp_token_idxs = token_idxs[start_idx:end_idx]
        expert_tokens = x[exp_token_idxs]
        expand_input.append(expert_tokens)
        if scales is not None:
            expert_scales = scales[exp_token_idxs]
            expand_scales.append(expert_scales)

    expert_first_token_offset = torch.Tensor(tokens_per_expert).to(torch.int64)

    expand_input = torch.cat(expand_input, dim=0)
    if recipe == "mxfp4":
        expand_input = expand_input.to(torch.uint8).view(torch.float4_e2m1fn_x2)
    else:
        expand_input = expand_input.to(data_dtype)
    expand_scales = (None if scales is None else torch.cat(expand_scales, dim=0).to(DEVICE))

    return expert_first_token_offset.to(DEVICE), expand_input.to(DEVICE), expand_scales


def ref_grouped_gemm(input_A, input_B, topk_ids, topk):
    num_experts = input_B.shape[0]
    dtype = input_A.dtype

    flat_expert_indices = topk_ids.view(-1)

    ref_expert_offset, ref_expand_input, ref_expand_scales = ref_prologue(input_A, None, flat_expert_indices, topk,
                                                                          num_experts, "bf16")

    ref = []
    pre_token_sum = 0
    for i in range(num_experts):
        cur_token_num = ref_expert_offset[i]
        if cur_token_num == 0:
            continue
        input = ref_expand_input[pre_token_sum:cur_token_num, :].to(torch.float32)
        weight = input_B[i, :, :].to(torch.float32)
        expert_output_fp32 = input @ weight
        ref.append(expert_output_fp32.to(dtype))
        pre_token_sum = cur_token_num
    ref = torch.cat(ref, dim=0)
    return ref


def get_fused_moe_benchmark(providers_filter: Optional[list[str]] = None, is_fp8=False, is_td_patched=False):
    supported_providers = {
        'triton' + ('-td' if is_td_patched else ''): 'triton' + ('-td' if is_td_patched else ''),
        'sycl-tla': 'sycl-tla',
    }

    providers = benchmark_suite.filter_providers(supported_providers, providers_filter)
    configs = MM_CONFIGS_FP8 if is_fp8 else MM_CONFIGS_BF16

    @benchmark_suite.perf_report(
        benchmark_suite.Benchmark(
            x_names=['num_tokens', 'output_hidden_size', 'hidden_size', 'num_experts', 'topk', 'dtype', 'has_bias'],
            x_vals=configs,
            line_arg='provider',
            line_vals=list(providers.keys()),
            line_names=list(providers.values()),
            styles=[('green', '-'), ('blue', '--'), ('red', ':')],
            ylabel=['GB/s', 'TFlops'],
            plot_name='fused-moe-gemm-performance' + ('-fp8' if is_fp8 else '') + ('-td' if is_td_patched else ''),
            args={},
        ))
    def benchmark(num_tokens, output_hidden_size, hidden_size, num_experts, topk, dtype, has_bias, provider):
        seed = 7
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        M = num_tokens
        N = output_hidden_size
        K = hidden_size
        use_fp8 = dtype == torch.float8_e4m3fn
        output_dtype = torch.bfloat16
        scores = torch.randn((M, num_experts), device=DEVICE, dtype=torch.float32)
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

        if use_fp8:
            hidden_states, input_A, input_scales = make_quantized_test_activations(
                1,
                M,
                K,
                in_dtype=output_dtype,
                quant_dtype=dtype,
            )
            hidden_states = hidden_states.squeeze(0)
            input_A = input_A.squeeze(0)
            input_scales = _normalize_fp8_scale(input_scales)

            input_B_orig, input_B, weight_scales, _ = make_test_weight(
                num_experts,
                N,
                K,
                in_dtype=output_dtype,
                quant_dtype=dtype,
            )
            weight_scales = _normalize_fp8_scale(weight_scales)
            input_B_ref = _dequantize_fp8(input_B, weight_scales).transpose(1, 2).contiguous()
            hidden_states_ref = _dequantize_fp8(input_A, input_scales)
        else:
            hidden_states = torch.randn((M, K), device=DEVICE, dtype=output_dtype) / 16
            input_A = hidden_states
            input_scales = None
            input_B_ref = torch.randn((num_experts, K, N), dtype=output_dtype, device=DEVICE)
            input_B = input_B_ref.transpose(1, 2).contiguous()
            weight_scales = None

        # Reference output
        output_ref = ref_grouped_gemm(
            input_A=(hidden_states_ref if use_fp8 else hidden_states).to(output_dtype),
            input_B=input_B_ref.to(output_dtype),
            topk_ids=topk_ids,
            topk=topk,
        )
        if use_fp8:
            # Free unused bf16 originals in the fp8 case.
            del hidden_states, input_B_orig

        m = M
        n = N
        k = K
        ws_shape = (m, topk, max(n, k))
        workspace = _resize_cache(
            torch.empty(
                ws_shape[0] * ws_shape[1] * ws_shape[2],
                dtype=output_dtype,
                device=input_A.device,
            ),
            (m, topk, n),
        )
        config = get_default_config(m, num_experts, n, k, topk, "fp8_w8a8" if use_fp8 else "bf16")
        sorted_token_ids, expert_ids, num_tokens_post_padded = torch_moe_align_block_size(
            topk_ids=topk_ids,
            block_size=config["BLOCK_SIZE_M"],
            num_experts=num_experts,
            pad_sorted_ids=True,
        )
        print(f"johnlu case M:{m}, num_experts:{num_experts}, N:{n}, K:{k}, topk: {topk} config:{config}")

        # Number of unique experts actually used in this batch
        num_activated_experts = topk_ids.unique().numel()
        # Total number of (token, expert) route pairs
        num_routed_tokens = m * topk
        print("johnlu sorted_token_ids:", sorted_token_ids.shape)
        print("johnlu sorted_token_ids:", sorted_token_ids)
        print("johnlu expert_ids:", expert_ids.shape)
        print("johnlu num_tokens_post_padded:", num_tokens_post_padded)

        quantiles = [0.5, 0.0, 1.0]
        n_warmup = 600

        if provider.startswith('triton'):

            def triton_fn():
                invoke_fused_moe_triton_kernel(
                    input_A,
                    input_B,
                    workspace,
                    input_scales,
                    weight_scales,
                    None,  # topk_weights
                    sorted_token_ids,  # sorted_token_ids
                    expert_ids,  # expert_ids
                    num_tokens_post_padded,  # num_tokens_post_padded
                    False,  # mul_routed_weights
                    topk,
                    config,
                    tl.bfloat16,
                    use_fp8,
                    False,
                    False,
                    False,
                    False,
                    None,
                    None,
                )
                return workspace

            triton_fn()  # workspace updated in-place
            output = workspace.clone().view(-1, n)
            num_output_tokens = m * topk
            valid_sorted_token_ids = sorted_token_ids[sorted_token_ids < num_output_tokens].to(torch.long)
            assert valid_sorted_token_ids.numel() == num_output_tokens
            output_triton_grouped = output[valid_sorted_token_ids]

            # Correctness check
            torch.testing.assert_close(
                output_triton_grouped,
                output_ref,
                rtol=6e-2 if use_fp8 else 2e-2,
                atol=6e-2 if use_fp8 else 1e-2,
            )

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                triton_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        elif provider == 'sycl-tla':
            # TODO: time SYCL-TLA's grouping/gather alongside the GEMM (via a native prologue) to match Triton's in-kernel gather and make the comparison fair.
            flat_expert_indices = topk_ids.view(-1)
            _, input_A_grouped, _ = ref_prologue(input_A, None, flat_expert_indices, topk, num_experts, "bf16")
            rows_per_expert = flat_expert_indices.bincount(minlength=num_experts).to(torch.int32).tolist()
            input_B_grouped = input_B.transpose(1, 2).contiguous()
            output_sycl = torch.empty((input_A_grouped.shape[0], n), dtype=input_A.dtype, device=DEVICE)

            def sycl_tla_fn():
                sycl_tla_grouped_gemm(input_A_grouped, input_B_grouped, None, output_sycl, rows_per_expert, n, k,
                                      num_experts)
                return output_sycl

            sycl_tla_fn()
            torch.testing.assert_close(output_sycl, output_ref, rtol=2e-2, atol=1e-2)

            _, min_ms, max_ms, mean_ms, cv = benchmark_suite.do_bench(
                sycl_tla_fn,
                n_warmup=n_warmup,
                n_repeat=10,
                quantiles=quantiles,
            )

        else:
            raise NotImplementedError(f'Unsupported provider {provider}')

        def gbps(ms):
            n_bytes = 1 if use_fp8 else 2
            total_bytes = (
                # B matrix: only load weights for activated experts
                num_activated_experts * K * N * n_bytes +
                # A matrix: each token is read once per expert assignment (topk times total)
                num_routed_tokens * K * n_bytes +
                # C matrix: output (one entry per routed token)
                num_routed_tokens * N * 2)
            return total_bytes * 1e-9 / (ms * 1e-3)

        def tflops(ms):
            # Each (token, expert) route pair performs a K×N matrix-vector product;
            # *2 accounts for the multiply-add operations in the matrix multiplication.
            total_flops = num_routed_tokens * N * K * 2
            return total_flops * 1e-12 / (ms * 1e-3)

        return (gbps(mean_ms), gbps(max_ms), gbps(min_ms)), (tflops(mean_ms), tflops(max_ms), tflops(min_ms)), cv

    return benchmark


def get_benchmark(providers_filter: Optional[list[str]] = None, is_fp8=False, is_td_patched=None):
    if is_td_patched is None:
        is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    return get_fused_moe_benchmark(
        providers_filter=providers_filter,
        is_fp8=is_fp8,
        is_td_patched=is_td_patched,
    )


if __name__ == '__main__':
    is_td_patched = os.getenv('TD_PATCHED', '0') == '1'
    print('Running fused MoE benchmark...')
    _benchmark_mm = get_fused_moe_benchmark(is_fp8=(os.getenv('FP8', '0') == '1'), is_td_patched=is_td_patched)
    _benchmark_mm.run(show_plots=False, print_data=True)
