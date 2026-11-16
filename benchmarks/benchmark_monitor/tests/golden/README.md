# Golden vectors for the benchmark_monitor regression gate

`cases.jsonl` holds 41 synthetic cases, one JSON object per line (schema: benchmark_monitor golden vectors v1). There is no real data. Each case has a series of `(run, night, value)`
observations, the resolved thresholds, `as_of`, and the expected verdict
`{status, drop_pct, z, baseline_median, mad, n_history}` rounded to 6 decimals.
`../test_golden.py` runs the gate on every case.

The expected values were produced by executing the reference rule from #6491 (`analysis.analyze`), not by hand.
`references` records what #6491 itself returns, per run and on per-night means, for the same input.

## Conventions

* `drop_pct` is a percent, positive = worse. `z` is negative = worse for higher-is-better metrics.
* `null` means the gate does not compute the value (`insufficient_history`, `noisy`, `missing`).
* `unit`: `night_mean` (default) averages the observations of one UTC night first. `run` keeps one point per run.
  The point with `night == as_of` is "tonight". Points after `as_of` are ignored, and so are observations with
  value <= 0.
* `references_disagree` is true when #6491 returns a different verdict than `expected`. `disagreeing_references`
  and `disagreement_cause` then say which reference and why. These are intended differences, for example catalog
  globs instead of exact benchmark names, per-night means instead of per-run values, and `missing` for a night
  without data.
* `fp_sensitive`: two cases sit within one ulp of z = -3. The gate must compute `0.6745 * (v - med) / mad` in that
  order, and the change as `(v - med) / med * 100`.
* The `lower-is-better-*` cases have no reference implementation. Their expected values mirror the reference gate
  (z and drop sign-flipped, lock-in disabled).

## Gate semantics pinned by the corpus

* The baseline is the series' own last `window` points before tonight, whatever their age.
* Lock-in follows #6491: dual median, recent window of 8, improvement of at least 8%, stability test of 5% on the
  recent MAD. The MAD falls back to 1% of the median.
* CV is the sample stdev over the mean, skipped only when strictly above `max_cv_pct`.
