# Golden vectors for the benchmark_monitor regression gate

`cases.jsonl` holds 47 synthetic cases, one JSON object per line (schema: benchmark_monitor golden vectors v1). There is no real data. Each case has a series of `(run, night, value)`
observations, the resolved thresholds, `as_of`, and the expected verdict
`{status, drop_pct, z, baseline_median, mad, n_history}` rounded to 6 decimals.
`../test_golden.py` runs the gate on every case.

The expected values were produced by executing the reference rule from #6491 (`analysis.analyze`), not by hand.
`references` records what #6491 itself returns, per run and on per-night means, for the same input.

## Conventions

* `drop_pct` is a percent, positive = worse. `z` is negative = worse for higher-is-better metrics.
* `null` means the gate does not compute the value (`insufficient_history`, `noisy`, `missing`).
* `unit`: `night_mean` (default) averages the observations of one UTC night first. `run` keeps one point per run.
  The point with `night == as_of` is "tonight". The corpus keeps observations dated after `as_of`; the gate itself
  must ignore them (`evaluate(night=as_of)`). Observations with value <= 0 are dropped before the gate, as the
  source layer does when it reads reports; the gate does not filter them.
* `references_disagree` is true when #6491 returns a different verdict than `expected`. `disagreeing_references`
  and `disagreement_cause` then say which reference and why. These are intended differences, for example catalog
  globs instead of exact benchmark names, per-night means instead of per-run values, and `missing` for a night
  without data.
* `fp_sensitive`: the verdict changes if an operation is evaluated in another order. The gate must compute
  `0.6745 * (v - med) / mad` and the change as `(v - med) / med * 100`, in those orders. `z-exactly-minus-3` and
  `z-exactly-plus-3` pin the z order; the `*-exactly-*pct` and `change-order-*` cases pin the change order
  (`100 * d / med`, `d * 100 / med` and `(v / med - 1) * 100` each flip at least one of them). The one-ulp z cases
  give the same verdict in every order and are not marked.
* No reference implements lower-is-better. In the `lower-is-better-*` cases the `pr6491_*` blocks are #6491 run on the
  same numbers with the direction mirrored (z and drop sign-flipped) and lock-in off. The gate does apply its lock-in
  to lower-is-better metrics, so `lower-is-better-lock-in` is a documented difference: its expected values come from
  the gate and were checked by hand.

## Gate semantics pinned by the corpus

* The baseline is the series' own last `window` points before tonight, whatever their age.
* Lock-in follows #6491: dual median, recent window of 8, improvement of more than 8%, stability test of 5% on the
  recent MAD. The MAD falls back to 1% of the median.
* CV is the sample stdev over the mean, skipped only when strictly above `max_cv_pct`.
