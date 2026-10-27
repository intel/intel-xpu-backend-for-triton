# Causal interior traversal with base-2 softmax

This branch independently starts from main `2ef1f41bb118631d0ca801895295a477b678beef` and updates
`unified_attention.patch` for vLLM pin
`3897f95cf17619137536d2a7d325d9c2f39dc38f`.
It preserves main's FP8 Q-scaling fix.

## Transformation

For plain causal BF16 2D TD attention with 64-token cache pages, process fully visible interior tiles without causal-mask construction, then process masked boundary tiles. Keep the running maximum in base-2 units across both loops and fuse exponent scaling/subtraction with FMA. Remove interior query-row predicates only for one query head per KV head. All other feature combinations retain the existing traversal and softmax.

## Port from the forum experiment

Source experiment: `experiment-83c61d9182854a6a`.

Main uses 16-wide BF16 tiles and groups heads using `num_queries_per_kv`. The specialization supports both 16- and 32-wide tiles without changing main's launch policy. The original one-head predicate-removal guard maps conservatively to `num_queries_per_kv == 1`; GQA still benefits from interior/boundary splitting and base-2 state while retaining its query predicates.

The forum's historical speedups were measured with its own launch configuration
and baseline. They are not performance measurements of this main-based port.

## Local validation

- The regenerated patch applies cleanly after the pinned dependency's common
  patches and reproduces the tested source files byte for byte.
- All affected Python files parse, and the expanded-source diff passes
  `git diff --check`.
- 17 focused correctness tests passed on Intel Arc B580 with
  PyTorch `2.16.0.dev20261006+xpu` and Triton `3.9.0`. The exact patched kernel
  and test bodies ran with lightweight vLLM import shims, using the existing
  independent PyTorch reference and its unchanged tolerances.
- Cases cover 16/32-wide tiles, causal interior/boundary regions, partial query
  rows, MHA/GQA, page-16 and padded-head cases, sliding windows, FP8 Q/K/V and
  the 3D fallback.

The focused regressions are in the existing patched test file. After normal
vLLM dependency preparation and TD-patch application, run:

```sh
python -m pytest -s --tb=short tests/kernels/attention/test_triton_unified_attention.py \
  -k 'td_causal_regions_and_tails or td_specialization_fallbacks'
```

Full vLLM CI and performance measurements on current main's compiler remain
to be run. No hosted workflow was dispatched as part of branch preparation.
