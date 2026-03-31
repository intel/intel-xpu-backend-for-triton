# vLLM XPU Test Integration

Scripts for installing and testing vLLM on Intel XPU as part of the Triton CI.

## Scripts

- **install-vllm.sh** - Clones, patches, and installs vLLM for XPU. Handles
  pinned commit checkout, `vllm-fix.patch` application, AST-based CUDA-to-XPU
  test patching, and dependency installation with torch constraint protection.
- **vllm-xpu-patch.py** - AST-guided patcher that scans vLLM test files for
  hardcoded CUDA references and applies source-level XPU replacements.

## Test Suites

### `--vllm-spec-decode` (TRITON_TEST_SUITE=vllm_spec_decode)

Validates speculative decoding Triton kernels on XPU.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `rejection_greedy_sample_kernel` | `vllm/v1/sample/rejection_sampler.py` | `tests/v1/sample/test_rejection_sampler.py` |
| `rejection_random_sample_kernel` | `vllm/v1/sample/rejection_sampler.py` | `tests/v1/sample/test_rejection_sampler.py` |
| `expand_kernel` | `vllm/v1/sample/rejection_sampler.py` | `tests/v1/sample/test_rejection_sampler.py` |
| `sample_recovered_tokens_kernel` | `vllm/v1/sample/rejection_sampler.py` | `tests/v1/sample/test_rejection_sampler.py` |
| `_prepare_eagle_inputs_kernel` | `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | `tests/v1/spec_decode/test_eagle.py` |
| `_update_eagle_inputs_kernel` | `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | `tests/v1/spec_decode/test_eagle.py` |

### `--vllm-mrv2` (TRITON_TEST_SUITE=vllm_mrv2)

Validates GPU Model Runner V2 Triton kernels on XPU.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `_gather_block_tables_kernel` | `vllm/v1/worker/gpu/block_table.py` | `tests/v1/worker/test_gpu_model_runner.py` |
| `_compute_slot_mappings_kernel` | `vllm/v1/worker/gpu/block_table.py` | `tests/v1/worker/test_gpu_model_runner.py` |
| `_apply_write_kernel` | `vllm/v1/worker/gpu/buffer_utils.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_prepare_prefill_inputs_kernel` | `vllm/v1/worker/gpu/input_batch.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_combine_sampled_and_draft_tokens_kernel` | `vllm/v1/worker/gpu/input_batch.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_post_update_kernel` | `vllm/v1/worker/gpu/input_batch.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_expand_idx_mapping_kernel` | `vllm/v1/worker/gpu/input_batch.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_get_num_sampled_and_rejected_kernel` | `vllm/v1/worker/gpu/input_batch.py` | `tests/v1/worker/test_gpu_input_batch.py` |
| `_temperature_kernel` | `vllm/v1/worker/gpu/sample/gumbel.py` | `tests/v1/sample/test_sampler.py` |
| `_gumbel_sample_kernel` | `vllm/v1/worker/gpu/sample/gumbel.py` | `tests/v1/sample/test_sampler.py` |
| `_topk_log_softmax_kernel` | `vllm/v1/worker/gpu/sample/logprob.py` | `tests/v1/sample/test_logprobs.py` |
| `_ranks_kernel` | `vllm/v1/worker/gpu/sample/logprob.py` | `tests/v1/sample/test_logprobs.py` |
| `_min_p_kernel` | `vllm/v1/worker/gpu/sample/min_p.py` | `tests/v1/sample/test_sampler.py` |
| `_penalties_kernel` | `vllm/v1/worker/gpu/sample/penalties.py` | `tests/v1/sample/test_sampler.py` |
| `_bincount_kernel` | `vllm/v1/worker/gpu/sample/penalties.py` | `tests/v1/sample/test_sampler.py` |
| `_prompt_logprobs_token_ids_kernel` | `vllm/v1/worker/gpu/sample/prompt_logprob.py` | `tests/v1/sample/test_logprobs.py` |
| `_bias_kernel` | `vllm/v1/worker/gpu/sample/logit_bias.py` | (via sampling e2e) |
| `_apply_grammar_bitmask_kernel` | `vllm/v1/worker/gpu/structured_outputs.py` | (via structured output e2e) |
| `_num_nans_kernel` | `vllm/v1/worker/gpu/metrics/logits.py` | `tests/v1/worker/test_gpu_model_runner.py` |

Both suites run with `VLLM_USE_V2_MODEL_RUNNER=1` to activate the V2 code paths.

### `--vllm-moe` (TRITON_TEST_SUITE=vllm_moe)

Validates MOE (Mixture of Experts) Triton kernels on XPU. Covers batched MOE,
fused MOE, DeepGemm MOE, and expert counting kernels.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `moe_mmk` | `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` | `tests/kernels/moe/test_batched_moe.py` |
| `expert_triton_kernel` | `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` | `tests/kernels/moe/test_batched_moe.py` |
| `batched_triton_kernel` | `vllm/model_executor/layers/fused_moe/fused_batched_moe.py` | `tests/kernels/moe/test_batched_moe.py` |
| `write_zeros_to_output` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | `tests/kernels/moe/test_moe.py` |
| `count_expert_num_tokens` | `vllm/model_executor/layers/fused_moe/utils.py` | `tests/kernels/moe/test_count_expert_num_tokens.py` |
| `fused_moe_kernel` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | `tests/kernels/moe/test_moe.py` |
| `fused_moe_kernel_gpta_awq` | `vllm/model_executor/layers/fused_moe/fused_moe.py` | `tests/kernels/moe/test_gpt_oss_triton_kernels.py` |
| `_silu_mul_fp8_quant_deep_gemm` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py` |
| `apply_expert_map` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |
| `_fwd_kernel_ep_scatter_1` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |
| `_fwd_kernel_ep_scatter_2` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |
| `_fwd_kernel_ep_gather` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |

### `--vllm-triton-attn` (TRITON_TEST_SUITE=vllm_triton_attn)

Validates Triton attention kernels on XPU (decode, prefill, unified, merge).

| Triton Kernel | Source File | Test File |
|---|---|---|
| `merge_attn_states_kernel` | `vllm/v1/attention/ops/triton_merge_attn_states.py` | `tests/kernels/attention/test_merge_attn_states.py` |
| `_fwd_kernel_stage1` | `vllm/v1/attention/ops/triton_decode_attention.py` | `tests/kernels/attention/test_triton_decode_attention.py` |
| `_fwd_grouped_kernel_stage1` | `vllm/v1/attention/ops/triton_decode_attention.py` | `tests/kernels/attention/test_triton_decode_attention.py` |
| `_fwd_kernel_stage2` | `vllm/v1/attention/ops/triton_decode_attention.py` | `tests/kernels/attention/test_triton_decode_attention.py` |
| `kernel_unified_attention_2d` | `vllm/v1/attention/ops/triton_unified_attention.py` | `tests/kernels/attention/test_triton_unified_attention.py` |
| `kernel_unified_attention_3d` | `vllm/v1/attention/ops/triton_unified_attention.py` | `tests/kernels/attention/test_triton_unified_attention.py` |
| `reduce_segments` | `vllm/v1/attention/ops/triton_unified_attention.py` | `tests/kernels/attention/test_triton_unified_attention.py` |

### `--vllm-gdn-attn` (TRITON_TEST_SUITE=vllm_gdn_attn)

Validates GDN (Gated Delta Net) attention kernels used by Qwen3-Next on XPU.
Runs in the `vllm-rest` CI job.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel` | `vllm/model_executor/layers/fla/ops/chunk_delta_h.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `chunk_fwd_kernel_o` | `vllm/model_executor/layers/fla/ops/chunk_o.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `chunk_scaled_dot_kkt_fwd_kernel` | `vllm/model_executor/layers/fla/ops/chunk_o.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `chunk_local_cumsum_*_kernel` | `vllm/model_executor/layers/fla/ops/cumsum.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `l2norm_fwd_kernel*` | `vllm/model_executor/layers/fla/ops/l2norm.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `layer_norm_fwd_kernel` | `vllm/model_executor/layers/fla/ops/layernorm_guard.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `solve_tril_16x16_kernel` | `vllm/model_executor/layers/fla/ops/solve_tril.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `merge_*_inverse_kernel` | `vllm/model_executor/layers/fla/ops/wy_fast.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |
| `recompute_w_u_fwd_kernel` | `vllm/model_executor/layers/fla/ops/wy_fast.py` | `tests/v1/attention/test_gdn_metadata_builder.py` |

### `--vllm-mamba` (TRITON_TEST_SUITE=vllm_mamba)

Validates Mamba SSM and causal conv1d Triton kernels on XPU. Also covers
mamba-mixer kernels (`mamba_ssm.py`, `ssd_*.py`) since they share the same test
files.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `_causal_conv1d_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | `tests/kernels/mamba/test_causal_conv1d.py` |
| `_causal_conv1d_update_kernel` | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | `tests/kernels/mamba/test_causal_conv1d.py` |
| `fused_gdn_gating_kernel` | `vllm/model_executor/layers/mamba/ops/gdn_linear_attn.py` | `tests/kernels/mamba/test_mamba_mixer2.py` |
| `_selective_scan_update_kernel` | `vllm/model_executor/layers/mamba/ops/mamba_ssm.py` | `tests/kernels/mamba/test_mamba_ssm.py` |
| `bmm_chunk_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_bmm.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |
| `chunk_scan_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |
| `chunk_cumsum_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_chunk_scan.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |
| `_chunk_state_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |
| `chunk_state_varlen_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_chunk_state.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |
| `state_passing_fwd_kernel` | `vllm/model_executor/layers/mamba/ops/ssd_state_passing.py` | `tests/kernels/mamba/test_mamba_ssm_ssd.py` |

### `--vllm-quant` (TRITON_TEST_SUITE=vllm_quant)

Validates quantization Triton kernels on XPU (INT8, FP8, AWQ, scaled matmul).
Runs in the `vllm-rest` CI job.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `scaled_mm_kernel` | `vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py` | `tests/kernels/quantization/test_triton_scaled_mm.py` |
| `awq_dequantize_kernel` | `vllm/model_executor/layers/quantization/awq_triton.py` | `tests/kernels/quantization/test_awq_triton.py` |
| `awq_gemm_kernel` | `vllm/model_executor/layers/quantization/awq_triton.py` | `tests/kernels/quantization/test_awq_triton.py` |
| `round_int8` | `vllm/model_executor/layers/quantization/utils/int8_utils.py` | `tests/kernels/quantization/test_int8_kernel.py` |
| `_per_token_quant_int8` | `vllm/model_executor/layers/quantization/utils/int8_utils.py` | `tests/kernels/quantization/test_int8_kernel.py` |
| `_per_token_group_quant_int8` | `vllm/model_executor/layers/quantization/utils/int8_utils.py` | `tests/kernels/quantization/test_block_int8.py` |
| `_w8a8_block_int8_matmul` | `vllm/model_executor/layers/quantization/utils/int8_utils.py` | `tests/kernels/quantization/test_block_int8.py` |
| `_per_token_group_quant_fp8` | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `tests/kernels/quantization/test_fp8_quant.py` |
| `_per_token_group_quant_fp8_colmajor` | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `tests/kernels/quantization/test_fp8_quant_group.py` |
| `_w8a8_block_fp8_matmul` | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | `tests/kernels/quantization/test_block_fp8.py` |

### `--vllm-linear-attn` (TRITON_TEST_SUITE=vllm_linear_attn)

Validates linear attention kernels (MiniMax-Text / Lightning Attention) on XPU.
Runs in the `vllm-rest` CI job.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `_fwd_diag_kernel` | `vllm/model_executor/layers/lightning_attn.py` | `tests/kernels/attention/test_lightning_attn.py` |
| `_fwd_kv_parallel` | `vllm/model_executor/layers/lightning_attn.py` | `tests/kernels/attention/test_lightning_attn.py` |
| `_fwd_kv_reduce` | `vllm/model_executor/layers/lightning_attn.py` | `tests/kernels/attention/test_lightning_attn.py` |
| `_fwd_none_diag_kernel` | `vllm/model_executor/layers/lightning_attn.py` | `tests/kernels/attention/test_lightning_attn.py` |
| `linear_attn_decode_kernel` | `vllm/model_executor/layers/lightning_attn.py` | `tests/kernels/attention/test_lightning_attn.py` |

### `--vllm-deepgemm` (TRITON_TEST_SUITE=vllm_deepgemm)

Validates DeepGemm-specific MOE kernels on XPU. Runs in the `vllm-rest` CI job.
Note: some test files overlap with `--vllm-moe`.

| Triton Kernel | Source File | Test File |
|---|---|---|
| `_silu_mul_fp8_quant_deep_gemm` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_silu_mul_fp8_quant_deep_gemm.py` |
| `apply_expert_map` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |
| `_fwd_kernel_ep_scatter_*` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_batched_deepgemm.py` |
| `_fwd_kernel_ep_gather` | `vllm/model_executor/layers/fused_moe/deep_gemm_utils.py` | `tests/kernels/moe/test_deepgemm.py` |

### `--vllm-kda` (TRITON_TEST_SUITE=vllm_kda)

KDA (Knowledge-Distilled Attention) kernels. No dedicated kernel-level tests
exist yet — only model-level integration. This suite is a placeholder and
currently prints a warning and skips.

**Kernel source:** `vllm/model_executor/layers/fla/ops/kda.py`

## CI Workflow

The CI workflow (`.github/workflows/vllm-tests.yml`) runs all suites in
parallel using a matrix strategy. Short suites are grouped into a single
`vllm-rest` job to reduce CI overhead:

| CI Matrix Entry | Test Suites Run |
|---|---|
| `vllm-spec-decode` | `--vllm-spec-decode` |
| `vllm-mrv2` | `--vllm-mrv2` |
| `vllm-moe` | `--vllm-moe` |
| `vllm-triton-attn` | `--vllm-triton-attn` |
| `vllm-mamba` | `--vllm-mamba` |
| `vllm-quant` | `--vllm-quant` |
| `vllm-rest` | `--vllm-gdn-attn`, `--vllm-linear-attn`, `--vllm-deepgemm` |

## Usage

```bash
# Install vLLM (requires pre-installed nightly torch/triton wheels)
bash scripts/vllm/install-vllm.sh --venv

# Run individual suites
bash scripts/test-triton.sh --vllm-spec-decode --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-mrv2 --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-moe --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-triton-attn --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-gdn-attn --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-mamba --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-quant --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-linear-attn --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --vllm-deepgemm --skip-pip-install --skip-pytorch-install
```

## TODO

- Populate skip lists from CI results (`scripts/skiplist/default/vllm_*.txt`)
- Refactor `run_vllm_install()` in `test-triton.sh` to delegate to `install-vllm.sh`
- Add dedicated kernel-level tests for KDA (`--vllm-kda`)
- Resolve pytest-shard/pytest-skip conflict properly (current workaround: uninstall pytest-shard)

## Reference

- vLLM PR [#36041](https://github.com/vllm-project/vllm/pull/36041) - Initial MRv2 CI tests for CUDA
- vLLM pinned commit: `benchmarks/vllm/vllm-pin.txt`
- XPU patch: `benchmarks/vllm/vllm-fix.patch`
