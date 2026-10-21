# Autotune pins

A pinned benchmark run gives every autotuner its config from here instead of autotuning, so each autotune key compiles
one config and benchmarks none. See `benchmarks/triton_benchmarks_pins.py`.

Layout: `<device id>/<benchmark>.json`, e.g. `b580/flash_attention.json`. The device id comes from the device name
(`Intel(R) Arc(TM) B580 Graphics` is `b580`).

## Modes

`AUTOTUNE_PINS_MODE` selects what `triton-benchmarks run` does:

- `record`: autotune as usual and append every decision, with the timings of all candidates, to
  `$AUTOTUNE_PINS_RECORDS/<benchmark>.jsonl`.
- `pinned`: use the pins in `$AUTOTUNE_PINS_DIR`. A key without a pin, or a pinned config that is no longer a
  candidate, is autotuned so that the run still measures it, and then fails the run.
- `auto`: `pinned` for benchmarks that have pins for the GPU, `record` for the others.

When a mode is set, reports have an `autotune_pins` column: `pinned@<pin file version>` or `full`.

## Updating

`triton-benchmarks-pins update` writes pin files from the records of full runs. It moves a pin only when another config
is faster by more than `--min-gain` (default 3%) and by more than the pin's own spread across the runs it is given. Give
it the records of several full runs to pick by median time.

```bash
AUTOTUNE_PINS_MODE=record AUTOTUNE_PINS_RECORDS=run1 triton-benchmarks run flash_attention
triton-benchmarks-pins update --records run1 --pins benchmarks/autotune_pins
AUTOTUNE_PINS_MODE=pinned AUTOTUNE_PINS_DIR=benchmarks/autotune_pins triton-benchmarks run flash_attention
```
