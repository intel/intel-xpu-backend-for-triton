# SPDX-License-Identifier: Apache-2.0
import random

import numpy as np
import torch
import triton.language as tl

from triton_grouped_gemm import (
    _resize_cache,
    invoke_fused_moe_triton_kernel,
    torch_moe_align_block_size,
    get_default_config,
)

DEVICE = "xpu"

RECIPE_TO_DTYPE = {
    "bf16": (torch.bfloat16, None),
    "fp16": (torch.float16, None),
    "mxfp8": (torch.float8_e4m3fn, torch.float8_e8m0fnu),
    "fp8block": (torch.float8_e4m3fn, torch.float32),
    "mxfp4": (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu),
}


def seed_everything(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def test_triton_grouped_gemm(input_A, input_B, topk_ids, topk):
    seed_everything(7)
    m = input_A.shape[0]
    k = input_A.shape[1]
    n = input_B.shape[-1]
    num_experts = input_B.shape[0]

    input_B = input_B.transpose(1, 2).contiguous()

    ws_shape = (m, topk, max(n, k))
    workspace = _resize_cache(
        torch.empty(
            ws_shape[0] * ws_shape[1] * ws_shape[2],
            dtype=input_A.dtype,
            device=input_A.device,
        ),
        (m, topk, n),
    )
    config = get_default_config(m, num_experts, n, k, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded = torch_moe_align_block_size(
        topk_ids=topk_ids,
        block_size=config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        pad_sorted_ids=True,
    )

    invoke_fused_moe_triton_kernel(
        input_A,
        input_B,
        workspace,
        None,  # input scales
        None,  # weight scales
        None,  # topk_weights
        sorted_token_ids,  # sorted_token_ids
        expert_ids,  # expert_ids
        num_tokens_post_padded,  # num_tokens_post_padded
        False,  # mul_routed_weights
        topk,
        config,
        tl.bfloat16,
        False,
        False,
        False,
        False,
        False,
        None,
        None,
    )

    output = workspace.clone().view(-1, n)
    # Triton writes C at flattened route positions (token_idx * topk + k_idx).
    # sorted_token_ids is padded with sentinel num_tokens for block alignment,
    # so keep only true (non-padding) route indices.
    num_tokens = m * topk
    valid_sorted_token_ids = sorted_token_ids[sorted_token_ids < num_tokens].to(torch.long)
    assert valid_sorted_token_ids.numel() == num_tokens
    output_triton_grouped = output[valid_sorted_token_ids]

    return output_triton_grouped


def test_grouped_gemm(m, n, k, e, topk, dtype, has_bias):
    seed_everything(7)
    num_experts = e
    scores = torch.randn((m, num_experts), device=DEVICE, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    hidden_states = torch.randn((m, k), device=DEVICE, dtype=torch.bfloat16) / 16

    # weight
    input_B = torch.randn((num_experts, k, n), dtype=dtype, device=DEVICE)

    output_triton = test_triton_grouped_gemm(input_A=hidden_states, input_B=input_B, topk_ids=topk_ids, topk=topk)
    output_ref = ref_grouped_gemm(input_A=hidden_states, input_B=input_B, topk_ids=topk_ids, topk=topk)

    torch.testing.assert_close(output_triton, output_ref, rtol=2e-2, atol=1e-2)


if __name__ == "__main__":
    print("Testing grouped GEMM on Xe...")
    print("Testing Qwen3-30B-A3B-Instruct w13 with MNK factors (80, 768 * 2 // 4, 2048), num_experts=128, topk=8")
    test_grouped_gemm(80, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False)
    print("Testing Qwen3-30B-A3B-Instruct w2 with MNK factors (80, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8")
    test_grouped_gemm(80, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False)
    print("Testing Qwen3-30B-A3B-Instruct w13 with MNK factors (8192, 768 * 2 // 4, 2048), num_experts=128, topk=8")
    test_grouped_gemm(8192, 768 * 2 // 4, 2048, 128, 8, torch.bfloat16, False)
    print("Testing Qwen3-30B-A3B-Instruct w2 with MNK factors (8192, 2048, 768 * 2 // 2 // 4), num_experts=128, topk=8")
    test_grouped_gemm(8192, 2048, 768 * 2 // 2 // 4, 128, 8, torch.bfloat16, False)
    print("Testing Llama-4-scout with MNK factors (30, 8192 * 2, 5120), num_experts=16, topk=1")
    test_grouped_gemm(30, 8192 * 2, 5120, 16, 1, torch.bfloat16, False)
    print("Testing Llama-4-scout with MNK factors (8192, 8192 * 2, 5120), num_experts=16, topk=1")
    test_grouped_gemm(8192, 8192 * 2, 5120, 16, 1, torch.bfloat16, False)
