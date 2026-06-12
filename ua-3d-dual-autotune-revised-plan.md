# Revised Implementation Plan: Separate Attention / Reduction Autotuners With Shared 3D Dependency Pruning

## Goal

Implement the initial unified-attention autotuning framework using two separate
Triton autotuners:

- one autotuner for `kernel_unified_attention`;
- one autotuner for `reduce_segments`.

The two kernels own separate config spaces and autotune caches. Values that are
local to one kernel may eventually have multiple candidates and be selected
independently. Values that affect the 3D producer/reducer contract must be
pruned to exactly one common value before either independent autotuner runs.

This first implementation tunes only `TILE_SIZE`. It does not tune
`NUM_SEGMENTS_PER_SEQ`, `num_warps`, `num_stages`, or any new block shape.
`num_stages=2` is fixed explicitly in every config to preserve the existing XPU
launch behavior.

## Assumptions And Scope

- No sentinel or active-segment metadata method is used.
- The attention kernel remains shared between the 2D and 3D paths.
- `TILE_SIZE` remains in the attention config space because 2D attention can
  choose between legal tile sizes.
- In 3D, `reduce_segments` reconstructs active segment count from `TILE_SIZE`,
  so producer and reducer must receive the same value.
- Removing or coordinating this dependency through a future architecture
  change is intentionally out of scope.

## Core Design

Keep the original two-config-space design:

```text
ATTENTION_AUTOTUNE_CONFIGS -> prune_attention_configs -> kernel_unified_attention
REDUCE_AUTOTUNE_CONFIGS    -> prune_reduce_configs    -> reduce_segments
```

Both prune functions use the same shared legality function. They then pass
through separate local-validation functions, which are intentionally empty in
the initial implementation but establish where future kernel-local rules belong.

```text
shared protocol validation
    + attention-local validation
    -> attention candidates

shared protocol validation
    + reducer-local validation
    -> reducer candidates
```

## Permanent Correctness Invariant

The reducer currently calculates:

```python
tiles_per_segment = cdiv_fn(
    seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE
)
act_num_segments = cdiv_fn(
    seq_len, tiles_per_segment * TILE_SIZE
)
```

Therefore, in 3D mode:

```text
attention TILE_SIZE == reduction TILE_SIZE
```

is a correctness requirement.

Under independent decorators, every dependency-bearing config must resolve to
exactly one common value before autotuning. Multiple surviving values are
unsupported unless the architecture changes to coordinate their selection.

For the current supported modes:

```text
3D BF16 -> TILE_SIZE=16
3D FP8  -> TILE_SIZE=32
```

This singleton rule is a permanent boundary of this design, not a temporary
first-step limitation. The framework remains extensible for attention-local and
reducer-local settings, while shared protocol settings remain singleton-pruned.

## Step 1: Define Separate Config Spaces

The implementation uses separate lists even though they initially contain the
same tile candidates:

```python
ATTENTION_AUTOTUNE_CONFIGS = [
    triton.Config({"TILE_SIZE": 16}, num_stages=2),
    triton.Config({"TILE_SIZE": 32}, num_stages=2),
]

REDUCE_AUTOTUNE_CONFIGS = [
    triton.Config({"TILE_SIZE": 16}, num_stages=2),
    triton.Config({"TILE_SIZE": 32}, num_stages=2),
]
```

Why separate lists:

- future attention-only candidates do not enter the reducer space;
- future reducer-only candidates do not enter the attention space;
- fixed launch metadata can diverge safely after benchmarking;
- shared config names remain visible in both spaces while the dependency exists.

Only `TILE_SIZE` varies. `num_stages=2` is explicit and fixed.

## Step 2: Define Keys Per Kernel

Attention preserves the existing key:

```python
ATTENTION_AUTOTUNE_KEY = [
    "BLOCK_SIZE",
    "HEAD_SIZE",
    "IS_FP8_INPUT",
    "IS_3D",
    "BLOCK_Q",
]
```

The reducer key contains only values needed to determine the legal shared tile:

```python
REDUCE_AUTOTUNE_KEY = [
    "BLOCK_SIZE",
    "IS_FP8_INPUT",
]
```

The reducer does not key on `HEAD_SIZE`, `BLOCK_Q`, or segment count yet because
no reducer-local performance setting is being selected. Those values can be
added if a future reducer-local config demonstrates that they affect its winner.

## Step 3: Define The Required Shared Config In One Place

Use one function that returns the complete required shared configuration rather
than one function per dependent value:

```python
DEPENDENT_3D_CONFIG_KEYS = ("TILE_SIZE",)


def required_3d_shared_config(is_fp8_input):
    """Return the single legal producer/reducer config for 3D attention."""
    return {"TILE_SIZE": 32 if is_fp8_input else 16}
```

If another dependency-bearing value is introduced later, it is added to this
dictionary and to `DEPENDENT_3D_CONFIG_KEYS`. Both kernels then consume the same
policy automatically.

## Step 4: Normalize Pruning Context

The current pruning rules need only block size, input type, and path:

```python
def _make_autotune_context(kwargs, *, is_3d):
    return {
        "block_size": kwargs["BLOCK_SIZE"],
        "is_fp8_input": kwargs["IS_FP8_INPUT"],
        "is_3d": is_3d,
    }
```

A dictionary is used instead of a dataclass to keep the first patch small. It
can become a typed object later if the context gains enough fields or behavior
to justify one.

## Step 5: Separate Shared And Local Validation

Shared validation enforces both ordinary tile legality and the permanent 3D
singleton contract:

```python
def _is_valid_shared_config(config, ctx):
    if ctx["is_3d"]:
        required = required_3d_shared_config(ctx["is_fp8_input"])
        for key in DEPENDENT_3D_CONFIG_KEYS:
            if key not in config.kwargs:
                raise ValueError(
                    f"Missing dependent 3D config value: {key}"
                )
            if config.kwargs[key] != required[key]:
                return False

    tile_size = config.kwargs["TILE_SIZE"]
    block_size = ctx["block_size"]

    if tile_size > block_size or block_size % tile_size != 0:
        return False
    if ctx["is_fp8_input"] and tile_size != 32:
        return False

    return True
```

The initial local validators deliberately accept every config:

```python
def _is_valid_attention_local_config(config, ctx):
    return True


def _is_valid_reduce_local_config(config, ctx):
    return True
```

They are included to establish a clear extension point. Future attention-only
or reducer-only rules should not be added to `_is_valid_shared_config`.

## Step 6: Implement Separate Prune Functions

One generic helper applies shared validation followed by kernel-local validation:

```python
def _prune_configs(configs, ctx, local_validator):
    return [
        config
        for config in configs
        if _is_valid_shared_config(config, ctx)
        and local_validator(config, ctx)
    ]


def prune_attention_configs(configs, named_args, **kwargs):
    ctx = _make_autotune_context(kwargs, is_3d=kwargs["IS_3D"])
    return _prune_configs(
        configs, ctx, _is_valid_attention_local_config
    )


def prune_reduce_configs(configs, named_args, **kwargs):
    ctx = _make_autotune_context(kwargs, is_3d=True)
    return _prune_configs(
        configs, ctx, _is_valid_reduce_local_config
    )
```

The reducer always uses `is_3d=True` because it is only launched after the 3D
producer path.

## Step 7: Add Independent Decorators

Attention:

```python
@triton.autotune(
    configs=ATTENTION_AUTOTUNE_CONFIGS,
    key=ATTENTION_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": prune_attention_configs},
)
@triton.jit
def kernel_unified_attention(...):
    ...
```

Reduction:

```python
@triton.autotune(
    configs=REDUCE_AUTOTUNE_CONFIGS,
    key=REDUCE_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": prune_reduce_configs},
)
@triton.jit
def reduce_segments(...):
    ...
```

In 3D, each decorator sees only one surviving tile, so Triton does not need to
benchmark competing shared values. The decorators remain independent, but the
shared policy guarantees that both receive the same tile.

In 2D BF16 attention, both 16 and 32 can survive when compatible with
`BLOCK_SIZE`, so the attention decorator performs a real selection.

## Step 8: Pass Reducer Pruning Metadata

`reduce_segments` receives two additional constexpr metadata values:

```python
BLOCK_SIZE: tl.constexpr
IS_FP8_INPUT: tl.constexpr
```

They are used by the reducer autotune key and prune callback, not its device
math. The launch changes from explicitly passing `TILE_SIZE_DECODE` to passing
the metadata needed for the reducer config to supply `TILE_SIZE`:

```python
reduce_segments[grid](
    ...,
    BLOCK_SIZE=block_size,
    IS_FP8_INPUT=is_fp8_input,
)
```

The explicit Python-side `TILE_SIZE_DECODE` variable is removed so the reducer
has one source of configuration.

## Step 9: Test The Singleton Contract Directly

For each supported 3D context, prune both spaces and compare their shared values:

```python
attention_tiles = {
    config.kwargs["TILE_SIZE"]
    for config in prune_attention_configs(...)
}
reduce_tiles = {
    config.kwargs["TILE_SIZE"]
    for config in prune_reduce_configs(...)
}

assert attention_tiles == reduce_tiles
assert len(attention_tiles) == 1
```

Required cases:

- BF16, `BLOCK_SIZE=16` -> both `{16}`;
- BF16, `BLOCK_SIZE=64` -> both `{16}`;
- FP8, `BLOCK_SIZE=64` -> both `{32}`;
- 2D BF16, `BLOCK_SIZE=64` -> attention retains `{16, 32}`;
- a 3D config missing `TILE_SIZE` raises an explicit error.

The implementation was checked against these cases with a focused host-side
test that executes the pruning helpers without launching a GPU kernel.

## Step 10: Correctness And Performance Validation

Apply the generated patch to the pinned vLLM source and run:

1. Python syntax validation.
2. Patch apply and reverse-apply checks.
3. Focused 3D correctness tests around tile and segment boundaries.
4. BF16 and FP8 benchmark comparisons against the existing attention-only
   autotune branch.
5. Repeated measurements for high-CV rows.

Boundary sequence lengths should include:

```text
15, 16, 17, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097
```

Because the reducer receives one legal tile in every 3D context, performance
changes in this first patch primarily measure decorator/configuration overhead
and any code-generation change caused by explicitly fixing `num_stages=2`.

## Step 11: Artifact Reporting

The decision artifact should eventually record separate fields for:

- attention selected config and `TILE_SIZE`;
- reducer selected config and `TILE_SIZE`;
- candidates remaining after each prune function;
- `attention_TILE_SIZE == reduce_TILE_SIZE`.

Artifact state must remain diagnostic only. The kernels must not coordinate
through mutable `best_config` state.

## Step 12: Extension Rules

Classify every new config before adding it:

```text
dependency-bearing:
  must be present in both config spaces and singleton-pruned to one common value

attention-local:
  may have multiple attention candidates and be selected independently

reducer-local:
  may have multiple reducer candidates and be selected independently
```

Examples of future local settings include attention or reducer `num_warps`,
fixed `num_stages` variants, and kernel-specific block shapes. A value such as
`NUM_SEGMENTS_PER_SEQ` remains dependency-bearing because it changes the
producer launch grid, buffer layout, and reducer interpretation.

Other backends can use the same structure while supplying different candidate
lists and `required_3d_shared_config` policy. The BF16/FP8 tile choices are XPU
policy, while the separation between shared and local configs is backend-neutral.

## Implemented Files

- `benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch`
  contains the dual-autotuner implementation.
  It also adds `tests/v1/attention/test_triton_unified_attention_autotune.py`
  to the patched vLLM tree for the singleton-pruning contract.
- `ua-3d-dual-autotune-revised-plan.md` documents the design and exact code.

## Initial Implementation Checklist

- [x] Separate attention and reducer config spaces.
- [x] Separate attention and reducer keys.
- [x] One shared required-config function.
- [x] Permanent singleton pruning for dependency-bearing 3D values.
- [x] Separate shared and kernel-local validation functions.
- [x] Attention and reducer autotune decorators.
- [x] Reducer metadata arguments and updated launch.
- [x] Direct host-side singleton-pruning checks.
- [x] Patch-application and Python syntax checks.
- [ ] GPU correctness tests.
- [ ] BF16 and FP8 performance benchmarks.
- [ ] Separate attention/reducer artifact fields.
