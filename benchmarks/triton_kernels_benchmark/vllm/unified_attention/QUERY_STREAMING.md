# Query-streaming BF16 attention candidate

This branch ports the locally validated query-streaming kernel into the existing
vLLM tensor-descriptor patch, based on main
`1b46be756df19968dd3ff4065fbd2ac56da0e773`. The vLLM pin remains
`3897f95cf17619137536d2a7d325d9c2f39dc38f`.

The QK loop streams 32 head dimensions at a time. P×V uses 32-column fragments
with partial loop unrolling, and attention scaling is fused into base-2 softmax.
The selected launch uses 128 query rows, eight subgroups, two stages and GRF256.
The JIT function's AST is unchanged from the measured research prototype.

Dispatch requires B580, `use_td=True`, BF16 Q/K/V/output, head dimension 128,
page size 16 or 64, contiguous buffers, plain causal attention and a maximum
query length above the split-K limit. It excludes batch-invariant execution,
split-K, per-sequence masks, sliding windows, softcap, bias, sinks, prefix/chunked
attention, descales and quantization. Other cases retain main's implementation.

## Local evidence

Two independent fresh-value/page-mapping confirmations of the prototype improved
all five workloads, with geometric-mean speedups of 1.194x and 1.193x over the
previous saved winner. The overall Triton/SYCL latency ratio fell from about
1.261 to 1.056; page16 still trails SYCL by 18–19%. These are measurements on the
local Torch 2.16.0.dev20261006+xpu / Triton 3.9.0 stack, not results of CI's fresh
compiler build.

| Q/KV heads | Batch | Sequence | Page | Previous winner ms | Candidate ms | SYCL ms |
|---|---:|---:|---:|---:|---:|---:|
| 28/4 | 1 | 4096 | 16 | 2.8971 | 2.7429 | 2.3157 |
| 28/4 | 4 | 2048 | 64 | 2.5291 | 1.9816 | 1.9877 |
| 32/8 | 1 | 4096 | 64 | 2.6191 | 2.2092 | 2.1075 |
| 32/8 | 4 | 2048 | 64 | 2.8739 | 2.3239 | 2.2508 |
| 64/4 | 1 | 4096 | 64 | 5.1861 | 4.2178 | 4.1103 |

The prototype passes the unchanged `atol=0.025, rtol=0.01` reference contract,
19 supplemental correctness cases and five matched VTune profiles with zero
spills. Fresh seeds use the same shapes; they do not establish broad performance
coverage.

The final applied branch patch additionally passes 21 focused B580 GPU tests:
eight fast-path tail/GQA cases and thirteen fallback cases. These use real
Torch/Triton and unchanged upstream test bodies/tolerances with lightweight
vLLM dependency import shims. A five-workload check of the public vLLM wrapper
passes independent correctness, preserves read-only inputs and produces exactly
the prototype's normalized native executable identity in every case, with zero
spills. Full vLLM installation and main's compiler build are left to CI.

The patch was generated against the prepared pinned vLLM sources. Forward and
reverse application checks reconstruct the tested source bytes. Existing FP8
fixes, test tolerances, benchmark inputs and workflow files are preserved.

## Benchmark workflow

Use `vllm-benchmarks.yml` with `runner_label=b580` and `tag=query-streaming`.
The existing workflow retains its BF16/FP8 unified-attention and MoE coverage,
including before/after patch measurements. Its results are needed to establish
broader behavior on the current compiler.
