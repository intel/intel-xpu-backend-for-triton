# vLLM XPU Test Integration

Scripts for installing and testing vLLM on Intel XPU as part of the Triton CI.

## Scripts

- **install-vllm.sh** - Clones, patches, and installs vLLM for XPU. Handles
  pinned commit checkout, `vllm-fix.patch` application, AST-based CUDA-to-XPU
  test patching, and dependency installation with torch constraint protection.
- **vllm-xpu-patch.py** - AST-guided patcher that scans vLLM test files for
  hardcoded CUDA references and applies source-level XPU replacements.

## Test Suites

Two test suites are registered in `test-triton.sh`:

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

## Usage

```bash
# Install vLLM (requires pre-installed nightly torch/triton wheels)
bash scripts/vllm/install-vllm.sh --venv

# Run spec decode tests
bash scripts/test-triton.sh --vllm-spec-decode --skip-pip-install --skip-pytorch-install

# Run MRv2 tests
bash scripts/test-triton.sh --vllm-mrv2 --skip-pip-install --skip-pytorch-install
```

## Reference

- vLLM PR [#36041](https://github.com/vllm-project/vllm/pull/36041) - Initial MRv2 CI tests for CUDA
- vLLM pinned commit: `benchmarks/vllm/vllm-pin.txt`
- XPU patch: `benchmarks/vllm/vllm-fix.patch`
