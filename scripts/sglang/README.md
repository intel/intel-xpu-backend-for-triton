# SGLang XPU tests

Install SGLang and run its Triton kernel tests on Intel XPU.

## Scripts

- `install-sglang.sh` - clones SGLang at `sglang-pin.txt`, applies
  `sglang-test-fix.patch`, uses `python/pyproject_xpu.toml`, drops
  `torch*`/`sglang-kernel-xpu`/`timm` from the requirements so the local torch
  and Triton survive, installs SGLang editable into `$TRITON_PROJ/sglang`
  without its Rust extensions.
- `sglang-pin.txt` - upstream commit.
- `sglang-test-fix.patch` - XPU fixes on top of the pin.
- `install-sgl-kernel-xpu.sh` - builds `sgl-kernel-xpu` at
  `sgl-kernel-xpu-pin.txt` and installs the wheel, providing the `sgl_kernel`
  module. **BMG/Xe2 only**; see `sgl_kernel` below.
- `sgl-kernel-xpu-pin.txt` - upstream `sgl-project/sgl-kernel-xpu` commit.

`import sglang` needs torchvision, which `install-sglang.sh` does not install.
CI builds it from `pytorch/.github/ci_commit_pins/vision.txt`.

## Suites

One flag per kernel family, each with its own `TRITON_TEST_SUITE` and skip list
`scripts/skiplist/<arch>/sglang_<family>.txt`. `--sglang` runs all of them.

| Flag | Test files, relative to `sglang/test/` |
|---|---|
| `--sglang-attention` | `registered/attention/test_create_kvindices.py`, `registered/attention/test_triton_attention_kernels.py`, `registered/attention/unittests/dense/test_triton.py`, `registered/kernels/ops/attention/test_fp4_indexer.py` (only with `sgl_kernel`) |
| `--sglang-quant` | `registered/kernels/ops/quantization/test_fp8_kernel.py`, `test_awq_dequant.py`, `registered/kernels/ops/gemm/test_fp8_kernel.py`, `test_triton_scaled_mm.py` |
| `--sglang-moe` | `registered/moe/test_fused_moe.py`, `registered/lora/test_fused_moe_lora_kernel.py` |
| `--sglang-mamba` | `registered/layers/mamba/test_causal_conv1d.py`, `test_mamba_ssm.py`, `test_mamba_ssm_ssd.py` |
| `--sglang-gdn` | `registered/attention/test_chunk_gated_delta_rule.py` |
| `--sglang-kda` | `registered/attention/test_kda_kernels.py` |
| `--sglang-spec` | `registered/spec/dspark/test_dspark_kernel_parity.py` |
| `--sglang-e2e` | `registered/xpu/test_xpu_basic.py` |

Three things to know before editing this:

- Suites run without `-n`. With `-n 4` the attention tests crash an xdist worker
  with a GPU page fault on a single-GPU runner.
- Skip list node ids start at `registered/`, not `test/registered/`: SGLang ships
  `test/pytest.ini`, so `sglang/test` is the pytest rootdir.
- `--sglang-quant` needs `--import-mode=importlib`: its two `test_fp8_kernel.py`
  have no `__init__.py`, so the default mode cannot collect both.

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
| `create_flashinfer_kv_indices_triton` | `kernels/ops/kvcache/kv_indices.py`, re-exported from `kernels/ops/attention/utils.py` |
| `get_num_kv_splits_triton` | `kernels/ops/attention/metadata.py` |
| `quantize_fp4_indexer_tensor`, `store_fp4_index_k_cache`, `fp4_index_logits_decode` (BMG only) | `kernels/ops/attention/dsv4/fp4_indexer.py` |

`unittests/dense/test_triton.py` is the second source of coverage for those
kernels and the only one that reaches `get_num_kv_splits_triton`. Instead of
calling the kernels directly it drives `RadixAttention` through the real
`TritonAttnBackend` and compares against HF-style torch reference modules, so it
also covers the metadata builders, page sizes 1/16/32, CUDA-graph decode,
split-op extend and spec-verify. Measured launches for one run: `_fwd_kernel` 55,
`_fwd_kernel_stage2` 41, `_fwd_kernel_stage1` 34, `create_flashinfer_kv_indices_triton` 70,
`get_num_kv_splits_triton` 22, `_fwd_grouped_kernel_stage1` 7.

Upstream gates it on `torch.cuda.is_available()` and registers it for CUDA and
ROCm only; `sglang-test-fix.patch` makes the gate, the RNG seeding and the
capture stream device-agnostic via `get_device_module()` and adds
`register_xpu_ci`. Those hunks are written to be upstreamable as-is.

`--sglang-quant`:

| Kernels | Source |
|---|---|
| `per_token_group_quant_fp8` | `kernels/ops/quantization/fp8_kernel.py` |
| `w8a8_block_fp8_matmul`, `triton_scaled_mm` | `kernels/ops/gemm/fp8_kernel.py` |
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
| `chunk_local_cumsum` | `kernels/ops/attention/fla/cumsum.py` |
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
| `dspark_accept`, `dspark_attn_metadata`, `dspark_draft_model`, `dspark_schedule`, `dspark_verify_window` | `kernels/ops/speculative/dspark/` |

`--sglang-e2e`. The only suite that runs a real forward pass, so the only one
that reaches these two - every other suite builds `ForwardBatch` directly and
passes `positions` in by hand, bypassing `ForwardBatch.init_new`:

| Kernels | Source |
|---|---|
| `compute_position_kernel` | `kernels/ops/attention/position.py` |
| `write_req_to_token_pool_triton` | `kernels/ops/memory/common.py` |

It also re-covers the attention kernels through the serving path. Measured
launches for one `bench_one_batch` run on Max 1100: `_fwd_grouped_kernel_stage1`
168, `_fwd_kernel_stage2` 168, `_fwd_kernel` 56, `create_flashinfer_kv_indices_triton`
8, `get_num_kv_splits_triton` 6, `compute_position_kernel` 2,
`write_req_to_token_pool_triton` 2.

Two things it does not give you. It asserts `decode_throughput > 0`, so it catches
a crash or a compile failure in those kernels but not a wrong value. And it needs
model weights (`Qwen/Qwen2.5-1.5B-Instruct`, ungated) plus a server launch, so it
is slower and less hermetic than the kernel suites - hence its own CI entry rather
than joining `sglang-rest`.

## Results on Max 1550

Local run at the current pin, one suite at a time. The skip lists come from it.

| Suite | Result | Time |
|---|---|---|
| `--sglang-attention` | 21 passed, 3 skipped (2 upstream, 1 skip-listed) | 24s |
| `--sglang-quant` | 5 passed | 14s |
| `--sglang-moe` | 110 passed | 46s |
| `--sglang-mamba` | 940 passed, 16 skipped upstream | 19s |
| `--sglang-gdn` | 30 passed, 1 skipped upstream | 9s |
| `--sglang-kda` | 1 passed, 14 skipped upstream | 10s |
| `--sglang-spec` | 1 skipped, skip-listed | 11s |
| `--sglang-e2e` | not measured locally - needs model weights and a server launch | - |

## Known gaps

- **`sgl_kernel`** (#8013), provided by `install-sgl-kernel-xpu.sh` on BMG only.
  Without it SGLang's `ModelRegistry` catches the `ImportError` and drops 208 of
  219 architectures, leaving 11 encoder/embedding models, so the e2e suite cannot
  load one: Qwen2 falls back to `TransformersForCausalLM`, which is dropped too.
  Three unguarded imports account for all of it, measured at pin `771e613d96`
  with the `activation.py` guard applied - `srt/layers/layernorm.py:102` (156
  models), `srt/layers/rotary_embedding/base.py:75` (161 once layernorm is
  fixed), `srt/layers/attention/vision.py:76` (41). Because of this the
  `sglang-e2e` matrix entry only exists where `install_sgl_kernel_xpu` is set:
  the suite passes on BMG and fails on PVC at `ValueError: Model architectures
  ['TransformersForCausalLM'] are not supported for now`. Guarding the three
  imports would let it run on PVC too, but `RMSNorm.forward_xpu` calls
  `rmsnorm`/`fused_add_rmsnorm` directly, so that needs real torch fallbacks
  rather than deferred import errors.

  `install-sglang.sh` strips the `sglang-kernel-xpu` requirement together with
  the `torch==2.13.0+xpu` pin next to it: the requirement is a prebuilt wheel
  linked against that torch, and the pin would replace ours. The out-of-band
  build avoids both (`--no-isolation`, and `dependencies = []` upstream, so
  nothing can replace torch or Triton).

  **No PVC.** `DPCPP_SYCL_TARGET` has no PVC value, and the kernel sources are
  architecture-gated in 79 places with no Xe-HPC branch, so there is nothing to
  target. vllm-xpu-kernels serves `max1100` and `b580` from one wheel
  (`SYCL_SUPPORTED_ARCHS` includes `intel_gpu_pvc`); this cannot.
  Upstream's only artifact is that wheel (`v0.3.0+xpu` in `sgl-project/whl`): it
  is not on PyPI (PyPI `sgl-kernel` is the unrelated CUDA package, the XPU
  distribution is `sglang-kernel-xpu`) and the `sgl-kernel-xpu` releases have no
  assets.

  A `bmg` wheel does load on PVC, and the light ops fall back to SPIR-V JIT
  correctly (on Max 1550: `rmsnorm` bit-exact, `silu_and_mul` 1.2e-03 in fp16,
  `hadamard_transform` runs). The CUTLASS-SYCL kernels do not: one
  `flash_attn_varlen_func` call ran 19m44s in the JIT without finishing. So
  installing it there would trade a loud `ImportError` for a hang.

  The build is slow - 871 ninja targets, 61m41s wall and 34 CPU-hours at `-j32`,
  7.5 GB of output, 83 `libsgl-ops-sycl-*.so`, dominated by the AOT CUTLASS-SYCL
  FMHA/MLA instantiations. `USE_SYCL_JIT=ON` is upstream's lever for that; it
  moves the cost to the first call per configuration and needs `icpx` at test
  time, which the CI shell already provides via `setvars.sh`.
- **Block pointers.** SGLang's fla kernels still call `tl.make_block_ptr`,
  removed from this Triton
  ([#7781](https://github.com/intel/intel-xpu-backend-for-triton/issues/7781)).
  `sglang-test-fix.patch` moves the GDN path, XPU overrides included, to tensor
  descriptors; the remaining uses, e.g. in `kda.py`, are not reached by the
  suites here.
- **CUDA-only tests.** `test_dspark_kernel_parity.py` calls `torch.cuda` and is
  skip-listed. Three of four `test_kda_kernels.py` classes skip themselves.
- **e2e is smoke only.** `--sglang-e2e` asserts throughput, not numerics, so a
  subtly wrong position id or token-pool write still passes. No SGLang test
  checks those two kernels against a reference.
- **Other `unittests/` families are still CUDA-gated.** Only `dense/test_triton.py`
  is enabled here. 29 more files under `registered/attention/unittests/` carry the
  same `torch.cuda.is_available()` gate, including the other Triton-backend rows
  `mla/test_triton.py`, `swa/test_triton.py`, `lightning/test_triton.py`,
  `kda/test_triton.py` and `gdn/test_triton.py`. Seven kits besides
  `speculative_draft_runner.py` also repeat the `torch.cuda` RNG pattern
  (`mla_attention.py`, `gdn_attention.py`, `kda_attention.py`, `mamba2_attention.py`,
  `lightning_attention.py`, `dsa_attention.py`, `dsv4_attention.py`). Enable one
  family at a time; `gdn/` and `kda/` stay blocked on `tl.make_block_ptr`
  regardless (see above).
- **Sliding window OOM.** `test_extend_attention_sliding_window` runs the kernel
  fine, but its torch reference needs more than 48 GB. Unskip when it is chunked.
- **BMG.** `scripts/skiplist/xe2/` is a copy of `default/`; nothing measured on
  B580 yet. `--skip-list` replaces the directory instead of merging, so the
  entries have to be duplicated.
- `install-sglang.sh` pins `xgrammar==0.2.7` (SGLang's CUDA manifest); every
  upstream XPU path pins `0.1.33`.

## CI

| Workflow | Trigger |
|---|---|
| `sglang-tests-reusable.yml` | `workflow_call`, builds the wheel once, then runs the suite matrix |
| `sglang-tests.yml` | `workflow_dispatch` with runner, pin, skip list and `install_sgl_kernel_xpu` overrides |
| `sglang-tests-pvc.yml` | Thursday/Sunday, `max1100` |
| `sglang-tests-bmg.yml` | Thursday/Sunday, `b60`, `skip_list: xe2`, `install_sgl_kernel_xpu: true` |
| `on-label.yml` | label `run-sglang-tests`, both PVC and BMG (BMG leg sets `install_sgl_kernel_xpu`) |

Matrix entries, one report artifact each, aggregated by the `reports` job:

| Entry | Runs |
|---|---|
| `sglang-attention` | `--sglang-attention` |
| `sglang-quant` | `--sglang-quant` |
| `sglang-moe` | `--sglang-moe` |
| `sglang-mamba` | `--sglang-mamba` |
| `sglang-rest` | `--sglang-gdn`, `--sglang-kda`, `--sglang-spec` |
| `sglang-e2e` | `--sglang-e2e` |

The short suites share `sglang-rest`, like `vllm-rest`. `sglang-e2e` gets its own
entry instead, because it launches a server and downloads model weights. Each
entry installs SGLang itself, because `run_sglang_tests` calls
`install-sglang.sh` - there is no install step in the workflow as there is for
vLLM.

`install_sgl_kernel_xpu` is only set for BMG. The setup job builds the
`sgl-kernel-xpu` wheel once and uploads it, and each matrix entry installs it
with `pip install --no-deps`. The wheel is cached in `/cache`, keyed by the
`sgl-kernel-xpu` pin, AOT target, PyTorch cache key and `icpx` version, so it is
only rebuilt when one of them changes.

## Usage

```bash
# needs torch and triton installed already
bash scripts/sglang/install-sglang.sh

# optional, BMG only: provides `sgl_kernel`. Needs oneAPI.
bash scripts/sglang/install-sgl-kernel-xpu.sh

bash scripts/test-triton.sh --sglang --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-attention --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-quant --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-moe --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-mamba --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-gdn --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-kda --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-spec --skip-pip-install --skip-pytorch-install
bash scripts/test-triton.sh --sglang-e2e --skip-pip-install --skip-pytorch-install
```

## Reference

- Issue [#7655](https://github.com/intel/intel-xpu-backend-for-triton/issues/7655)
  - the agreed kernel and test list
- Issue [#8013](https://github.com/intel/intel-xpu-backend-for-triton/issues/8013)
  - the `sgl_kernel` gap
- SGLang RFC #29630 - the `sglang.kernels` namespace
- `sgl-project/sgl-kernel-xpu` - the SYCL kernel library, built from source
