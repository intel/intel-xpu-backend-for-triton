# vLLM Triton kernel coverage report (#8349 + #8422)

_Date: 2026-10-30_

## Repo state
- Both PRs are merged: #8349 on 10-14 and #8422 on 10-20. vLLM CI on main is green on PVC and BMG.
- Main has moved on since then, to `fe357dd0b`. It has a newer vLLM pin (#8476) and a few follow-up PRs, e.g. #8508, which unskipped the KDA tests.

## Results

**Kernels** (176 `kernel` rows in `vllm-kernels-snapshot.xlsx`, plus `fused_moe_kernel`, which was already covered):

| | Covered |
|---|---|
| Before #8349 | 35 / 176 (20%) |
| After #8422 | **126 / 176 (72%)** |
| Newly covered | **+91** (none lost) |
| Device helpers | about 40 / 59 (counted when a covered kernel calls them) |

**Tests** (PVC nightly):

| | Passed | Skipped | Failed |
|---|---|---|---|
| Before (10-14) | 13,270 | 3,932 | 0 |
| After our PRs (10-21) | 20,238 | 4,766 | 0 |
| Change | **+6,968 (+53%)** | **+834** | 0 |
| Today (10-29, with later PRs) | 21,331 | 4,623 | 0 |

- **Where the new passes come from:**
  - 4 new suites: lora (4,686), model_ops (91), batch_invariant (58) and v1_runner (4).
  - Growth in existing suites: triton_attn +858, moe +531, quant +263, mamba +204, mrv2 +175, spec_decode +79, kda +19.
- **BMG:** 10,496 → 21,370 passing today. The 10-21 BMG run is a poor comparison point because runner outages left some suites out.
- **Skiplists:** 20 entries were added on PVC and 19 on BMG. They are grouped by root cause, and 7 tracking issues are still open (#8351, #8353, #8355, #8356, #8357, #8361, #8406).
- **Skipped count:** most of the +834 are test parametrizations that skip themselves on XPU (e.g. CUDA-only dtypes or backends). Only a small share are skiplist entries.
- **Guards:** 79 CUDA-only guards across 37 test files were relaxed. Each one hides a test that launches a listed kernel and runs on XPU unmodified.

## Why 50 kernels are still not covered

| Reason | Count |
|---|---|
| No upstream vLLM test reaches the kernel | 33 |
| Only reached through a full model run or with more than 1 GPU | 11 |
| Tests exist but never reach the kernel; covering it would mean changing the test | 5 |
| A test reaches it, but hangs on XPU (>40 min, `test_draft_model.py`) | 1 |

The biggest gap is simply missing upstream tests. We don't write our own tests, so those 33 kernels stay uncovered until vLLM adds tests for them.

**No upstream test (33):** `kv_cache_gather_kernel`, `kv_cache_scatter_kernel`, `_fused_moe_lora_kernel`,
`fused_olmo_hybrid_gdn_gating_kernel`, `_bailing_linear_attn_decode_spec_step_kernel`, `_layer_norm_fwd_1pass_kernel`,
`_minimax_qk_var_kernel`, `_minimax_rms_apply_kernel`, `_vit_mrope_setup_kernel`, `_bilinear_pos_embed_kernel`,
`FusedMTPInputRMSNormKernel.kernel`, `MTPSharedHeadRMSNormKernel.kernel`, `SavePartialStatesKernel.kernel`,
`_xpu_qnorm_rope_kernel`, `_dequant_gather_slots_kernel`, `_apply_swa_score_mask_kernel`, `_gather_swa_kv_kernel`,
`_inkling_gate_select_kernel`, `_qkvr_qkv_kernel`, `_sink_epilogue_kernel`, `chunk_gla_fwd_kernel_o`,
`kda_gate_chunk_cumsum_vector_kernel`, `recompute_w_u_fwd_kernel`, `chunk_kda_fwd_kernel_inter_solve_fused`,
`chunk_kda_fwd_kernel_intra_sub_chunk`, `chunk_kda_fwd_kernel_intra_token_parallel`, `fused_recurrent_kda_fwd_kernel`,
`fused_recurrent_kda_packed_decode_kernel`, `_num_nans_kernel`, `_min_p_kernel`, `_bincount_kernel`,
`_penalties_kernel`, `_prompt_logprobs_token_ids_kernel`

**Needs a model checkpoint or >1 GPU (11):** `_fill_short_context_topk_indices`,
`ComputeGlobalTopkIndicesAndLensKernel.kernel`, `_add_rmsnorm_fwd_kernel`, `_cache_draft_logits_kernel`,
`_selector_walk_kernel`, `_cache_inputs_kernel`, `_pad_trailing_draft_slots_kernel`, `_prepare_input_buffers_kernel`,
`_prepare_input_hidden_states_and_embeddings_kernel`, `_shift_input_embeds_kernel`, `_shift_input_ids_kernel`

**Tests never reach the kernel (5):** `_fused_kv_compress_norm_rope_insert_sparse_attn`,
`PrepareUniformDecodeKernel.kernel`, `ComputeDSparkNoncausalSWAIndicesKernel.kernel`,
`reshape_and_cache_kernel_flash_diffkv`, `_tq_full_dequant_kv`

**Hangs on XPU (1):** `_prepare_prefill_inputs_kernel`

## Limitations
- **Blind spot in the measurement:** the kernel recorder can't see launches in vLLM's engine subprocess. About 117 end-to-end test files were therefore under-mapped. A re-map with `VLLM_ENABLE_V1_MULTIPROCESSING=0` (about 11 GPU-hours) may move some of the 11 "needs a model" kernels into the covered column.
- **Point-in-time numbers:** coverage was measured at the vLLM pin of the time. New pins can add or drop coverage; for example, the newer pin added 2 kernels.
- **"Covered" means launched at least once:** we picked a minimal set of tests per kernel, so a covered kernel may be exercised by only a few shapes or dtypes.
- **Skiplisted tests still count as covered:** a kernel whose only test is skiplisted is still collected by the suite but isn't really verified until its issue is fixed.
- **BMG runner instability:** the B60 runners often lose connection, so BMG results needed re-runs and day-to-day comparisons there are noisy.

## Possible next steps
1. Re-map the end-to-end tests with multiprocessing off.
2. Debug the `test_draft_model.py` hang.
3. Work through the open skip issues.
