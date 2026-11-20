# IGC reproducer: fp8 unified attention slowdown on IGC 2.43.9 / agama 40734.8 (Arc B580)

Standalone reproducer for https://github.com/intel/intel-xpu-backend-for-triton/issues/8934.
It has **no Triton, torch or SYCL dependency**: it feeds the SPIR-V that Triton generated for vLLM's
`kernel_unified_attention` straight to the Level Zero driver (which invokes IGC) and times it.

Needs: `g++`, `libze_loader` + Level Zero headers (`level_zero/ze_api.h`), and `ocloc`. Run:

    ./run.sh                      # ITERATIONS=50 WARMUP=10 OCLOC_DEVICE=bmg by default

## What it does, per case and per build variant

Two variants of the same SPIR-V are built, because Triton picks between them at run time (it first builds with the
default GRF mode, and rebuilds with `-cl-intel-256-GRF-per-thread` if the zebin reports any register spill):

* `default`: no build flags.
* `grf256`:  `-cl-intel-256-GRF-per-thread`. This is what Triton uses for this kernel on IGC 2.42.6.

For each variant it prints the `ocloc` result (GRF count, spill size), then launches the kernel through Level Zero and prints the
median/min device-timestamp time, TFLOPS and a checksum of the output buffer. The compiled zebins are kept in `results/`
so they can be diffed between IGC versions. **Compare the same package on an old-IGC and a new-IGC machine**: the checksums must match
(same results), the times should not.

## Cases

* `prefill4096`: one 4096x4096 causal sequence, 64 q-heads / 8 kv-heads, head size 128, block size 64, fp8 e4m3 Q/K/V. In CI this is
  the largest drop (about 0.59x with the new stack).
* `decode8`: eight decode sequences (q_len 1, kv_len 34..4803), same head config. Only the main kernel is replayed; the small
  `reduce_segments` kernel that follows it in Triton is not.

Inputs: Q/K/V are random fp8 bytes (NaN encodings avoided) from a fixed xorshift; the block table, sequence lengths and descales are
shipped verbatim (`*.bin`) because they steer which KV blocks are read. Attention has no data dependent control flow beyond that.

## Baseline numbers (IGC 2.42.6, agama 40196.8, Arc B580, this package)

| case | variant | spill | p50 ms | TFLOPS |
|---|---|---|---|---|
| prefill4096 | default | 14144 B | 70.3 | 7.8 |
| prefill4096 | grf256 | 320 B | 31.9 | 17.2 |
| decode8 | default | 11776 B | 0.390 | 1.2 |
| decode8 | grf256 | 320 B | 0.291 | 1.6 |

Cross-check on the same host: the same call through Triton takes 32.3 ms (prefill4096) and 0.30 ms (decode8, including `reduce_segments`).
CI measured about 16.9 TFLOPS for prefill4096 on the old stack, so the replay is faithful for the prefill case. The decode case is short
(0.24-0.29 ms for grf256 across runs on this host), so its timing is noisier; use it for old-vs-new comparisons on one machine type and repeat it.

## Provenance

* SPIR-V captured with `provenance/capture.py` (vLLM at the CI pin `692cf2d75ae765ed08ec9ccc075f46a1c09a98f1`, unpatched kernel, no TD) on a
  B580 with Triton 3.9.1+gitb9defb36 (the venv build, not the exact CI commit `0ae12ba`) and the old driver stack. The generated code
  is therefore from the old stack's Triton; only the driver compiler differs between the runs this package is meant to compare.
* `gen_case.py <capture-dir> <case-dir>` turns a capture into a case directory (`case.cfg` + `kernel.spv` + index tensors).
* `provenance/time_triton.py` times the same call through Triton for the cross-check above.

## Open questions this package is meant to answer

1. On IGC 2.43.9, does the `grf256` build get slower, or does the `default` build change so that Triton no longer rebuilds with 256 GRF?
   (`ocloc` spill for `default` is the signal: Triton rebuilds only when it is non-zero.)
2. If `grf256` itself is slower, diff `results/*_grf256.bin` between the two IGC versions.
