# SGLang XPU tests

Install SGLang and run its Triton kernel tests on Intel XPU.

## Scripts

- `install-sglang.sh` - clones SGLang at `sglang-pin.txt`, applies
  `sglang-test-fix.patch`, uses `python/pyproject_xpu.toml`, drops
  `torch*`/`sgl-kernel`/`timm` from the requirements so the local torch and
  Triton survive, installs SGLang editable into `$TRITON_PROJ/sglang`.
- `sglang-pin.txt` - upstream commit.
- `sglang-test-fix.patch` - XPU fixes on top of the pin.

`import sglang` needs torchvision, which `install-sglang.sh` does not install.
CI builds it from `pytorch/.github/ci_commit_pins/vision.txt`.

## Suites

One flag per kernel family, each with its own `TRITON_TEST_SUITE` and skip list
`scripts/skiplist/<arch>/sglang_<family>.txt`. `--sglang` runs all of them.

| Flag | Test files, relative to `sglang/test/` |
|---|---|
| `--sglang-attention` | `registered/attention/test_create_kvindices.py`, `registered/attention/test_triton_attention_kernels.py` |
| `--sglang-quant` | `registered/quant/test_fp8_kernel.py`, `test_triton_scaled_mm.py`, `test_awq_dequant.py` |
| `--sglang-moe` | `registered/lora/test_fused_moe_lora_kernel.py` |
| `--sglang-mamba` | `registered/layers/mamba/test_causal_conv1d.py`, `test_mamba_ssm.py`, `test_mamba_ssm_ssd.py` |
| `--sglang-gdn` | `registered/attention/test_chunk_gated_delta_rule.py` |
| `--sglang-kda` | `registered/attention/test_kda_kernels.py` |
| `--sglang-spec` | `registered/spec/dspark/test_dspark_kernel_parity.py` |

Two things to know before editing this:

- Suites run without `-n`. With `-n 4` the attention tests crash an xdist worker
  with a GPU page fault on a single-GPU runner.
- Skip list node ids start at `registered/`, not `test/registered/`: SGLang ships
  `test/pytest.ini`, so `sglang/test` is the pytest rootdir.

## Kernel coverage

Kernel paths are relative to `sglang/python/sglang/`. SGLang moved all Triton
kernels to `kernels/ops/` (RFC #29630), so recheck them after a pin bump.

`--sglang-attention`, the highest-value suite - the first four rows are what
`srt/layers/attention/triton_backend.py` imports:

| Kernels | Source |
|---|---|
| `decode_attention_fwd`, `_normal`, `_grouped` | `kernels/ops/attention/decode_attention.py` |
| `extend_attention_fwd`, `_unified`, `build_unified_kv_indices` | `kernels/ops/attention/extend_attention.py` |
| `context_attention_fwd` | `kernels/ops/attention/prefill_attention.py` |
| `create_flashinfer_kv_indices_triton` | `kernels/ops/attention/utils.py` |

`--sglang-quant`:

| Kernels | Source |
|---|---|
| `per_token_group_quant_fp8`, `w8a8_block_fp8_matmul`, `triton_scaled_mm` | `kernels/ops/quantization/fp8_kernel.py` |
| `awq_dequantize`, `awq_gemm` | `kernels/ops/quantization/awq_triton.py` |

`--sglang-moe`:

| Kernels | Source |
|---|---|
| `fused_moe_lora` | `kernels/ops/moe/fused_moe_lora_kernel.py` |

`--sglang-mamba`. causal conv1d is shared with GDN and KDA, so this suite covers
those paths too:

| Kernels | Source |
|---|---|
| `causal_conv1d_fn`, `causal_conv1d_update` | `kernels/ops/mamba/causal_conv1d_triton.py` |
| `selective_state_update` | `kernels/ops/mamba/triton_ops/mamba_ssm.py` |
| `mamba_chunk_scan_combined`, chunk-cumsum / chunk-state / chunk-scan / state-passing / bmm chain, `chunk_state_varlen` | `kernels/ops/mamba/triton_ops/ssd_*.py` |

`--sglang-gdn`, the only SGLang Triton test upstream registers for XPU
(`register_xpu_ci(est_time=900)`):

| Kernels | Source |
|---|---|
| `chunk_gated_delta_rule` | `kernels/ops/attention/fla/chunk.py` |
| `chunk_gated_delta_rule_fwd_h` | `kernels/ops/attention/fla/chunk_delta_h.py` |
| `chunk_gated_delta_rule_fwd_intra` | `kernels/ops/attention/fla/chunk_fwd.py` |
| `chunk_fwd_o` | `kernels/ops/attention/fla/chunk_o.py` |
| `chunk_scaled_dot_kkt_fwd` | `kernels/ops/attention/fla/chunk_scaled_dot_kkt.py` |
| `chunk_local_cumsum` | `kernels/ops/attention/fla/cumsum.py` |
| `solve_tril` | `kernels/ops/attention/fla/solve_tril.py` |
| `recompute_w_u_fwd` | `kernels/ops/attention/fla/wy_fast.py` |
| `fused_recurrent_gated_delta_rule` | `kernels/ops/attention/fla/fused_recurrent.py` |

`fla/chunk.py` and `fla/kda.py` reroute two of these to
`srt/hardware_backend/xpu/kernels/fla/` under `if is_intel:`. The detector reads
`triton.runtime.driver.active.get_current_target().backend` and swallows
`BaseException`, falling back to `"cpu"` - a target-reporting regression in the
fork silently picks the NVIDIA kernels instead of failing.

`--sglang-kda`:

| Kernels | Source |
|---|---|
| `fused_recurrent_kda`, `kda_gate_chunk_cumsum`, `chunk_kda_scaled_dot_kkt_fwd` | `kernels/ops/attention/fla/kda.py` |
| `fused_recurrent_kda_packed_decode` | `kernels/ops/attention/fla/fused_recurrent.py` |
| `fused_sigmoid_gating_delta_rule_update` | `kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py` |
| `chunk_local_cumsum` | `kernels/ops/attention/fla/cumsum.py` |

`--sglang-spec`:

| Kernels | Source |
|---|---|
| `pad_verify_lens_to_bucket`, `build_qo_indptr` | `kernels/ops/speculative/ragged_verify_kernels.py` |
| `expand_prefill_causally`, `build_page_table_positions`, `build_causal_swa_page_indices` | `kernels/ops/attention/dsv4_attn_metadata_kernels.py` |
| `dspark_accept`, `dspark_attn_metadata`, `dspark_draft_model`, `dspark_schedule`, `dspark_verify_window` | `srt/speculative/dspark_components/kernels/` |

## Results on Max 1100

Local run at the current pin, one suite at a time. The skip lists come from it.

| Suite | Result | Time |
|---|---|---|
| `--sglang-attention` | 8 passed, 2 skipped (1 upstream, 1 skip-listed) | 26s |
| `--sglang-quant` | 5 passed | 43s |
| `--sglang-moe` | 108 skipped, all skip-listed | 4s |
| `--sglang-mamba` | 932 passed, 16 skipped upstream | 15s |
| `--sglang-gdn` | 29 skipped, all skip-listed | 4s |
| `--sglang-kda` | 1 passed, 12 skipped upstream | 6s |
| `--sglang-spec` | 1 skipped, skip-listed | 6s |

Nothing failed because of Triton codegen.

## Known gaps

- **sgl-kernel.** Five files from #7655 cannot be imported without it, so they
  are in no suite: `test_fp4_indexer.py` (imports `sgl_kernel` directly) plus
  `test_int8_kernel.py`, `test_block_int8.py`, `test_fused_moe.py` and
  `test/manual/test_triton_moe_wna16.py` (all import
  `srt/layers/activation.py`, whose XPU branch needs it). Wire them in once
  sgl-kernel-xpu is installable in CI.
- **Block pointers.** All 29 GDN tests and most of KDA fail to compile: SGLang's
  fla kernels still call `tl.make_block_ptr`, removed from this Triton
  ([#7595](https://github.com/intel/intel-xpu-backend-for-triton/issues/7595)).
  The XPU overrides use it too, so the fix has to come from SGLang. XPU signal
  on the one test upstream registers for XPU is zero until then.
- **CUDA-only tests.** `test_fused_moe_lora_kernel.py` is parametrized with
  `device="cuda:0"` and `test_dspark_kernel_parity.py` calls `torch.cuda`; both
  are skip-listed. Two of three `test_kda_kernels.py` classes skip themselves.
- **Sliding window OOM.** `test_extend_attention_sliding_window` runs the kernel
  fine, but its torch reference needs more than 48 GB. Unskip when it is chunked.
- **BMG.** `scripts/skiplist/xe2/` is a copy of `default/`; nothing measured on
  B580 yet. `--skip-list` replaces the directory instead of merging, so the
  entries have to be duplicated. The B580 callers pass `max_jobs: "8"` because
  the default `-j(2 * cores)` OOM-kills cc1plus on that runner.
- `install-sglang.sh` pins `xgrammar==0.2.1` (SGLang's CUDA manifest); every
  upstream XPU path pins `0.1.33`.

## CI

| Workflow | Trigger |
|---|---|
| `sglang-tests-reusable.yml` | `workflow_call`, builds the wheel once, then runs the suite matrix |
| `sglang-tests.yml` | `workflow_dispatch` with runner, pin and skip list overrides |
| `sglang-tests-pvc.yml` | Thursday/Sunday, `max1100` |
| `sglang-tests-bmg.yml` | Thursday/Sunday, `b580`, `skip_list: xe2` |
| `on-label.yml` | label `run-sglang-tests`, both PVC and BMG |

Matrix entries, one report artifact each, aggregated by the `reports` job:

| Entry | Runs |
|---|---|
| `sglang-attention` | `--sglang-attention` |
| `sglang-quant` | `--sglang-quant` |
| `sglang-moe` | `--sglang-moe` |
| `sglang-mamba` | `--sglang-mamba` |
| `sglang-rest` | `--sglang-gdn`, `--sglang-kda`, `--sglang-spec` |

The short suites share `sglang-rest`, like `vllm-rest`. Each entry installs
SGLang itself, because `run_sglang_tests` calls `install-sglang.sh` - there is no
install step in the workflow as there is for vLLM.

## Usage

```bash
# needs torch and triton installed already
bash scripts/sglang/install-sglang.sh

bash scripts/test-triton.sh --sglang --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-attention --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-quant --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-moe --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-mamba --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-gdn --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-kda --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-spec --skip-pip-install --skip-pytorch-install
```

## Reference

- Issue [#7655](https://github.com/intel/intel-xpu-backend-for-triton/issues/7655)
  - the agreed kernel and test list
- SGLang RFC #29630 - the `sglang.kernels` namespace
