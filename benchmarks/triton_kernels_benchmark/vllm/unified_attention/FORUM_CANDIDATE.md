# Page-scoped K/V descriptor reuse

This branch independently starts from main `2ef1f41bb118631d0ca801895295a477b678beef` and updates
`unified_attention.patch` for vLLM pin
`3897f95cf17619137536d2a7d325d9c2f39dc38f`.
It preserves main's FP8 Q-scaling fix.

## Transformation

Create the K and V descriptors once per 64-token physical cache page in BF16 2D TD attention, then load its intersecting subtiles. Keep separate K/V base addresses and strides, clip partial page intervals and avoid page-table loads for empty intervals. Preserve main's natural-log softmax; the inherited exponent-fusion change from the forum experiment is deliberately excluded.

## Port from the forum experiment

Source experiment: `experiment-680e148d6e104cf3`.

Main uses 16-wide BF16 tiles, whereas the forum experiment used 32-wide tiles. Derive subtiles per page as `BLOCK_SIZE // TILE_SIZE` and support both widths without changing launch settings. This is the isolated descriptor transformation, not a claim to reproduce the combined forum experiment's speedup.

The forum's historical speedups were measured with its own launch configuration
and baseline. They are not performance measurements of this main-based port.

## Local validation

- The regenerated patch applies cleanly after the pinned dependency's common
  patches and reproduces the tested source files byte for byte.
- All affected Python files parse, and the expanded-source diff passes
  `git diff --check`.
- 21 focused correctness tests passed on Intel Arc B580 with
  PyTorch `2.16.0.dev20261006+xpu` and Triton `3.9.0`. The exact patched kernel
  and test bodies ran with lightweight vLLM import shims, using the existing
  independent PyTorch reference and its unchanged tolerances.
- Cases cover 16/32-wide tiles, causal interior/boundary regions, partial query
  rows, MHA/GQA, page-16 and padded-head cases, sliding windows, FP8 Q/K/V and
  the 3D fallback.
- Additional cases cover clipped page intervals, softcap and different K/V strides.

The focused regressions are in the existing patched test file. After normal
vLLM dependency preparation and TD-patch application, run:

```sh
python -m pytest -s --tb=short tests/kernels/attention/test_triton_unified_attention.py \
  -k 'td_causal_regions_and_tails or td_specialization_fallbacks or td_page_clipping_and_distinct_strides'
```

Full vLLM CI and performance measurements on current main's compiler remain
to be run. No hosted workflow was dispatched as part of branch preparation.
