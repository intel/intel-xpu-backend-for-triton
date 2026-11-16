# benchmark_monitor

Nightly benchmark regression monitoring for Triton benchmark results. It reads measurements, aggregates them
into one point per night, and gates each series against its own recent history.

## Install

```
pip install -e benchmarks/benchmark_monitor
```

Runtime dependency: `pyyaml`. Python 3.10 or newer.

## The catalog

`benchmark_monitor/catalog/default.yaml` is the single place that says what is monitored and how strictly:

- `defaults`, `families` (benchmark-name globs), `benchmarks` and `platforms` set the gate thresholds. They are
  resolved in that order; within `families` the first match in declaration order wins.
- `metrics` declares every metric the monitor may see and whether higher is better. A metric with `gate: false`
  is recorded but never gated. A metric that is not declared is an error, never a guess.
- `scopes` choose the series to monitor (source, platform, tags, benchmark globs).

`catalog/schema.json` describes the file for editors. A site can add platforms and scopes through overlays, but an
overlay cannot change thresholds.

## Tests

```
python -m pytest benchmarks/benchmark_monitor/tests
```
