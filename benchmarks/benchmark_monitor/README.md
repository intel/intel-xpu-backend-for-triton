# benchmark_monitor

Nightly benchmark regression monitoring for Triton benchmark results. It reads measurements, aggregates them
into one point per night, and gates each series against its own recent history.

This part holds the data model (`benchmark_monitor.model`) and the aggregation of raw observations into one
point per UTC night, or one per run (`benchmark_monitor.aggregate`). Stdlib only.

## Install

```
pip install -e benchmarks/benchmark_monitor
```

Python 3.11 or newer.

## Tests

```
python -m pytest benchmarks/benchmark_monitor/tests
```
