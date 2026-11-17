# benchmark_monitor

Nightly benchmark regression monitoring for Triton benchmark results. It reads measurements, aggregates them
into one point per night, and gates each series against its own recent history.

## Install and run

```
pip install -e benchmarks/benchmark_monitor
python -m benchmark_monitor --artifacts DIR --scope nightly-b580-micro --as-of 2026-11-16 --format md
```

`DIR` is searched recursively for `*-report.csv` files (wide or long format). Other flags: `--catalog PATH`
(default: the shipped catalog), `--format md|json`, `--out PATH`, `--history-days N` (calendar days read before `--as-of`; default 3 x the largest
catalog window, 0 = all history; the gate's `window` counts points, so gaps need a wider span). Exit code 2 means a bad argument or catalog.
Runtime dependency: `pyyaml`; Python 3.10 or newer.

## The catalog

`benchmark_monitor/catalog/default.yaml` says what is monitored and how strictly:

- `defaults`, `families` (benchmark-name globs), `benchmarks` and `platforms` set the gate thresholds, resolved in
  that order. Within `families` the first match in declaration order wins.
- `metrics` declares every metric the monitor may see and whether higher is better. `gate: false` records a
  metric without gating it. An undeclared metric is skipped with one warning, never guessed.
- `scopes` choose the series to monitor (source, platform, tags, benchmark globs).

`catalog/schema.json` describes the file for editors. Overlays can add platforms and scopes but cannot change
thresholds.

## Sources

Anything that yields `Observation` objects for a scope and a date range can feed the monitor, for example a
database-backed source: implement the `Source` protocol in `benchmark_monitor.source`.

## Tests

```
python -m pytest benchmarks/benchmark_monitor/tests
```
