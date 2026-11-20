# Golden vectors for the benchmark_monitor regression gate

`cases.jsonl` holds 62 synthetic cases, one JSON object per line (schema: benchmark_monitor golden vectors v1). There is no real data. Each case has a series of `(run, night, value)`
observations, the resolved thresholds, `as_of`, and the expected verdict
`{status, drop_pct, z, baseline_median, mad, n_history}` rounded to 6 decimals.
`../test_golden.py` runs the gate on every case.

The expected values were produced by executing the reference rule from #6491 (`analysis.analyze`), not by hand.
`references` records what #6491 itself returns, per run and on per-night means, for the same input.

## Conventions

* `drop_pct` is a percent, positive = worse. `z` is sign-adjusted like `drop_pct`: negative = worse for both
  higher-is-better and lower-is-better metrics.
* `null` means the gate does not compute the value (`insufficient_history`, `noisy`, `missing`).
* `unit`: `night_mean` (default) averages the observations of one UTC night first. `run` keeps one point per run.
  The point with `night == as_of` is "tonight". The corpus keeps observations dated after `as_of`; the gate itself
  must ignore them (`evaluate(night=as_of)`). Observations with value <= 0 are dropped before the gate, as the
  source layer does for a metric that is not declared `signed: true` (reports write 0 for a metric a row did
  not measure); every corpus metric is unsigned. The gate does not filter them.
* An observation's `night` is a date (midnight UTC) or an ISO timestamp. An offset is honoured and a timestamp without
  one is UTC; the UTC calendar date of the instant is the night (`utc-*` cases: 23:30 and 00:30 UTC, non-UTC offsets,
  and `unit` `run` ordering by time).
* `references_disagree` is true when #6491 returns a different verdict than `expected`. `disagreeing_references`
  and `disagreement_cause` then say which reference and why. These are intended differences, for example catalog
  globs instead of exact benchmark names, per-night means instead of per-run values, and `missing` for a night
  without data.
* `fp_sensitive` (tag `fp`, kept in sync by the test): the verdict changes if an operation is evaluated in another
  order. With `delta = sign * (v - med)` (`sign` is +1 for higher-is-better, -1 otherwise) the gate must compute
  `z = 0.6745 * delta / mad` and the change as `delta / abs(med) * 100` (that is `sign * (v - med) / abs(med) * 100`),
  in those orders. `z-exactly-minus-3` and `z-exactly-plus-3` pin the z order; the `*-exactly-*pct` and
  `change-order-*` cases pin the change order (`100 * d / med`, `d * 100 / med` and `(v / med - 1) * 100` each flip at
  least one of them). The one-ulp z cases give the same verdict in every order and are not marked.
* No reference implements lower-is-better. In the `lower-is-better-*` cases the `pr6491_*` blocks are #6491 run on the
  same numbers with the direction mirrored (z and drop sign-flipped) and lock-in off. The gate does apply its lock-in
  to lower-is-better metrics, so `lower-is-better-lock-in` is a documented difference: its expected values come from
  the gate and were checked by hand.

## Gate semantics pinned by the corpus

* The baseline is the series' own last `window` points before tonight, whatever their age.
* Lock-in follows #6491: dual median; the recent window is the last `max(min_history, min(lock_window, n))` points
  (8 by default); the baseline locks to the recent level only if its median is more than `improvement_lock_pct`
  (8%) above the full median and the recent MAD is strictly below `lock_mad_frac` (5%) of the recent median. The MAD
  falls back to 1% of the median.
* #6491 fixes the lock window at 8 and the stability fraction at 0.05. `lock-in-window-12-wired` and
  `lock-in-mad-frac-10pct-wired` use other values: their expected values come from #6491 with that one constant taken
  from the threshold, and their `pr6491_*` blocks are the unmodified #6491, so they are documented differences.
* The thresholds are read from the case, not fixed: the `threshold`-tagged cases use non-default `z`, `max_cv_pct`,
  `window`, `min_history`, `lock_window`, `lock_mad_frac` and `improvement_lock_pct`.
* CV is the sample stdev over the absolute mean, in percent. A series is `noisy` only when CV is strictly above
  `max_cv_pct`; the check is skipped when the mean is 0.
