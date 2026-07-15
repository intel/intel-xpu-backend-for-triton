# Unified-attention `BLOCK_Q` autotune-pruning fix

## Change metadata

- Branch: `fix/ua-td-block-q-pruning`
- Remote-main base: `2b6c7559081ef6d73e6d92d38d51fa81205fda96`
- Implementation commit: `648116db9f47faa7db5bcb7d1bfa26d7e9cdac56`
- Changed implementation file:
  `benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch`

The tracked file is itself a patch applied to vLLM's
`vllm/v1/attention/ops/triton_unified_attention.py`. The code below is
shown as it appears in the effective patched vLLM source, without the extra
leading `+` characters used by the outer patch file.

## Problem

The original autotune pruner required every candidate-derived `BLOCK_Q` to
be a power of two:

```python
def prune_attention_configs(configs, named_args, **kwargs):
    return [
        config
        for config in configs
        if _is_power_of_2(
            derive_block_q(config.kwargs["BLOCK_M"], kwargs["num_queries_per_kv"])
        )
    ]
```

That condition was applied regardless of the active memory-access path.
However, `BLOCK_Q` is a tensor-descriptor `block_shape` dimension only
for Q loads and output stores, which are enabled by `USE_TD_QO`.

The K/V tensor descriptors enabled by `USE_TD` use:

```python
desc = tl.make_tensor_descriptor(
    base=base,
    shape=(BLOCK_SIZE, HEAD_SIZE),
    strides=(stride_cache_1, stride_cache_3),
    block_shape=(TILE_SIZE, HEAD_SIZE_PADDED),
)
```

They do not include `BLOCK_Q`. Consequently, pointer Q/O and K/V-TD-only
executions may use a non-power-of-two `BLOCK_Q`.

| `USE_TD` | `USE_TD_QO` | K/V access | Q/O access | `BLOCK_Q` requirement |
|---:|---:|---|---|---|
| `False` | `False` | Pointer | Pointer | Positive; may be non-power-of-two |
| `True` | `False` | Tensor descriptor | Pointer | Positive; may be non-power-of-two |
| `True` | `True` | Tensor descriptor | Tensor descriptor | Positive and power-of-two |
| `False` | `True` | Invalid/unreachable | — | — |

## Complete effective code before the change

The following includes the surrounding configuration, helper functions,
autotune decorator, heuristic, and kernel boundary.

```python
ATTENTION_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": block_m}, num_stages=2)
    for block_m in (16, 32, 64)
]

ATTENTION_AUTOTUNE_KEY = [
    "BLOCK_SIZE",
    "HEAD_SIZE",
    "IS_3D",
    "num_queries_per_kv",
]


def derive_block_q(block_m, num_queries_per_kv):
    return block_m // num_queries_per_kv


def _is_power_of_2(value):
    return value > 0 and triton.next_power_of_2(value) == value


def prune_attention_configs(configs, named_args, **kwargs):
    return [
        config
        for config in configs
        if _is_power_of_2(
            derive_block_q(config.kwargs["BLOCK_M"], kwargs["num_queries_per_kv"])
        )
    ]


@triton.autotune(
    configs=ATTENTION_AUTOTUNE_CONFIGS,
    key=ATTENTION_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": prune_attention_configs},
)
@triton.heuristics(
    {
        "BLOCK_Q": lambda args: derive_block_q(
            args["BLOCK_M"], args["num_queries_per_kv"]
        )
    }
)
@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path. In 3D mode per-segment partials
    # go to the segment tensors.
    output_ptr,
    # Remaining kernel arguments follow.
    ...
):
    ...
```

## Complete effective code after the change

```python
ATTENTION_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": block_m}, num_stages=2)
    for block_m in (16, 32, 64)
]

ATTENTION_AUTOTUNE_KEY = [
    "BLOCK_SIZE",
    "HEAD_SIZE",
    "IS_3D",
    "num_queries_per_kv",
    "USE_TD",
    "USE_TD_QO",
]


def derive_block_q(block_m, num_queries_per_kv):
    return block_m // num_queries_per_kv


def _is_power_of_2(value):
    return value > 0 and triton.next_power_of_2(value) == value


def prune_attention_configs(configs, named_args, **kwargs):
    all_args = {**named_args, **kwargs}
    num_queries_per_kv = all_args["num_queries_per_kv"]
    use_td_qo = all_args.get("USE_TD_QO", False)

    valid_configs = []
    for config in configs:
        block_q = derive_block_q(
            config.kwargs["BLOCK_M"], num_queries_per_kv
        )

        if block_q <= 0:
            continue
        if use_td_qo and not _is_power_of_2(block_q):
            continue

        valid_configs.append(config)

    return valid_configs


@triton.autotune(
    configs=ATTENTION_AUTOTUNE_CONFIGS,
    key=ATTENTION_AUTOTUNE_KEY,
    prune_configs_by={"early_config_prune": prune_attention_configs},
)
@triton.heuristics(
    {
        "BLOCK_Q": lambda args: derive_block_q(
            args["BLOCK_M"], args["num_queries_per_kv"]
        )
    }
)
@triton.jit
def kernel_unified_attention(
    # Output destination for the 2D path. In 3D mode per-segment partials
    # go to the segment tensors.
    output_ptr,
    # Remaining kernel arguments follow.
    ...
):
    ...
```

## Exact executable changes

### 1. Distinguish execution paths in the autotune key

These two entries were added:

```diff
 ATTENTION_AUTOTUNE_KEY = [
     "BLOCK_SIZE",
     "HEAD_SIZE",
     "IS_3D",
     "num_queries_per_kv",
+    "USE_TD",
+    "USE_TD_QO",
 ]
```

This prevents pointer, K/V-TD-only, and full Q/K/V/O-TD invocations from
sharing an autotune result merely because their tensor dimensions match.
Those paths compile different kernel bodies and can have different valid or
optimal configurations.

### 2. Make pruning conditional on `USE_TD_QO`

```diff
 def prune_attention_configs(configs, named_args, **kwargs):
-    return [
-        config
-        for config in configs
-        if _is_power_of_2(
-            derive_block_q(config.kwargs["BLOCK_M"], kwargs["num_queries_per_kv"])
+    all_args = {**named_args, **kwargs}
+    num_queries_per_kv = all_args["num_queries_per_kv"]
+    use_td_qo = all_args.get("USE_TD_QO", False)
+
+    valid_configs = []
+    for config in configs:
+        block_q = derive_block_q(
+            config.kwargs["BLOCK_M"], num_queries_per_kv
         )
-    ]
+
+        if block_q <= 0:
+            continue
+        if use_td_qo and not _is_power_of_2(block_q):
+            continue
+
+        valid_configs.append(config)
+
+    return valid_configs
```

The resulting rules are:

1. `BLOCK_Q <= 0` is invalid for every path.
2. A positive, non-power-of-two `BLOCK_Q` remains available for pointer Q/O.
3. A positive, non-power-of-two `BLOCK_Q` is rejected when
   `USE_TD_QO=True`.
4. `USE_TD=True, USE_TD_QO=False` does not impose a power-of-two
   `BLOCK_Q` constraint because only K/V use descriptors in that mode.

## Unchanged Q/O descriptor eligibility checks

`num_queries_per_kv` and `head_size` are fixed invocation inputs, not
autotuned configuration fields. Their Q/O descriptor restrictions therefore
remain in the wrapper outside the pruning function:

```python
head_size_padded = triton.next_power_of_2(head_size)
_is_pow2_nq = (num_queries_per_kv & (num_queries_per_kv - 1)) == 0
_is_pow2_hs = head_size == head_size_padded
use_td_qo = use_td and _is_pow2_nq and _is_pow2_hs
```

If either fixed input is unsuitable, `USE_TD_QO` becomes false and Q/O
fall back to masked pointer operations. K/V may continue using tensor
descriptors when `use_td` remains true.

## Nested patch metadata changes

Expanding the first inserted vLLM hunk by 13 lines required updating its
target count and every subsequent target-side hunk location in the same
embedded patch. These are patch bookkeeping changes; they do not change the
code contained in those later hunks.

| Original hunk header | Updated hunk header |
|---|---|
| `@@ -175,6 +173,49 @@` | `@@ -175,6 +173,62 @@` |
| `@@ -290,6 +331,9 @@` | `@@ -290,6 +344,9 @@` |
| `@@ -534,12 +578,12 @@` | `@@ -534,12 +591,12 @@` |
| `@@ -561,27 +605,14 @@` | `@@ -561,27 +618,14 @@` |
| `@@ -699,7 +730,6 @@` | `@@ -699,7 +743,6 @@` |
| `@@ -709,7 +739,7 @@` | `@@ -709,7 +752,7 @@` |
| `@@ -929,11 +959,6 @@` | `@@ -929,11 +972,6 @@` |
| `@@ -949,22 +974,9 @@` | `@@ -949,22 +987,9 @@` |
| `@@ -975,12 +987,16 @@` | `@@ -975,12 +1000,16 @@` |
| `@@ -1036,18 +1052,11 @@` | `@@ -1036,18 +1065,11 @@` |
| `@@ -1068,17 +1077,57 @@` | `@@ -1068,17 +1090,57 @@` |
| `@@ -1150,9 +1199,7 @@` | `@@ -1150,9 +1212,7 @@` |
| `@@ -1183,7 +1230,6 @@` | `@@ -1183,7 +1243,6 @@` |

No source-side hunk positions changed.

## Validation performed

The implementation commit was checked as follows:

- `git diff --check` passed.
- The updated embedded patch passed `git apply --check --verbose` against
  the prepared vLLM source pinned by remote main.
- Both patched Python files passed `python3 -m py_compile`.
- Focused pruning assertions covered:

| Scenario | Expected retained `BLOCK_M` values | Result |
|---|---|---|
| Pointer path, `num_queries_per_kv=3` | `16, 32, 64` | Passed |
| K/V TD with pointer Q/O, `num_queries_per_kv=3` | `16, 32, 64` | Passed |
| Full Q/O TD, `num_queries_per_kv=4` | `16, 32, 64` | Passed |
| Any path with `BLOCK_Q=0` candidate | Candidate rejected | Passed |
| Synthetic `BLOCK_M=24`, `num_queries_per_kv=4`, pointer Q/O | `BLOCK_Q=6` retained | Passed |
| Synthetic `BLOCK_M=24`, `num_queries_per_kv=4`, full Q/O TD | `BLOCK_Q=6` rejected | Passed |
