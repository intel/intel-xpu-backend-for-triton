# Query-row predicate removal

This branch independently starts from main `2ef1f41bb118631d0ca801895295a477b678beef` and updates
`unified_attention.patch` for vLLM pin
`3897f95cf17619137536d2a7d325d9c2f39dc38f`.
It preserves main's FP8 Q-scaling fix.

## Transformation

Remove query-validity predicates from the score mask for plain causal BF16 TD attention, while retaining the causal sequence mask and the output-store bounds. The fast path requires 2D execution, TD Q/O, one query head per KV head and an unpadded head dimension. Optional attention features and quantization retain the existing path.

## Port from the forum experiment

Source experiment: `experiment-493958541ca3481e`.

The forum kernel has a separate `heads_per_program` launch dimension. Main groups heads using `num_queries_per_kv`, so the conservative one-head guard maps to `num_queries_per_kv == 1`. This port therefore optimizes MHA; GQA remains on the existing path. It does not import the forum launch dispatcher or offline tuning tables.

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
