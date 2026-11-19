# Offline UA decision trees

`learn.py` trains a cost-sensitive configuration tree from a JSON timing matrix.
The package includes the generic optimizer, a UA adapter, a standalone policy
interpreter, and Python/JSON exporters. It runs on the CPU with Python's standard
library; neither PyTorch nor an ML package is needed for training or selection.
The generated selectors can be integrated into a kernel's host dispatch.

```bash
python scripts/vllm/ua_tuning/learn.py path/to/measurements.json \
  --output-dir reports/ua-tree --max-depth 6 --leaf-penalty 0.002
```

Multiple measurement files can be positional inputs. Each needs its adjacent
`.metadata.json`; an adjacent `.manifest.json` is retained in provenance when
available. Direct tuner metadata and the consolidated sweep metadata are both
supported. Mixing GPUs, driver/compiler/kernel versions, or timing methods is
rejected. Workload IDs must be unique across inputs.

The output contains:

- `policy.json`: the deployable combined policy: one shared config pool, exact
  measured anchors, and a learned tree for exact misses.

- `selector.py`: ordinary Python branches and a dictionary of complete configs.
- `inline_selector.py`: the same branches with `_UA_TUNED_CONFIGS` as a literal
  module-level constant and UA-prefixed helpers, ready to embed in the UA module.
- `decisions.py`: `DECISIONS`, mapping condition tuples to complete configs,
  plus a selector with the same API and domain/legality checks as `selector.py`.
- `decisions.json`: portable named decision entries, each with a list of
  conditions and a complete config, the domain/target contract, and source hashes.
- `tree.json`: versioned tree, all config definitions, feature/domain contract,
  target environment, input/source hashes, and exact-search statistics.
- `report.json`: per-workload selections, normalized regret, coverage, greedy
  comparison, the best shared config when one exists, and the feature-oracle bound.

## Dictionary decisions

Each dictionary entry describes one leaf of the learned tree. Conditions within
its key are combined with AND; every entry contains its complete config:

```python
DECISIONS = {
    (("query_work", "<=", 68880.0), ("num_queries_per_kv", "<=", 7.5)): {
        "block_m": 16, "tile_size": 16, "num_warps": 4,
        "num_stages": 2, "grf_mode": "256",
    },
    # Other learned regions follow.
}
```

Right branches become `>` for numerical splits and `!=` for categorical splits.
Separate regions remain separate entries even when they choose the same config.
A tree with one leaf has the empty condition tuple `()` as its dictionary key.
JSON uses names such as `leaf_000` with explicit condition records because JSON
object keys cannot be tuples.

```python
from decisions import DECISIONS, select_config
config = select_config(device_name=device_name, **features)
```

Selection evaluates the conditions and retains the learned threshold behavior.
The dictionary selector has the same support checks and heuristic-fallback
contract as the branch-based selector. All Python exports are checked against
the JSON tree interpreter.

## Inline constants in the UA module

`inline_selector.py` follows the constant-dictionary-plus-heuristics approach:
`_UA_TUNED_CONFIGS` holds each used complete config once, and generated Python
branches select its dictionary entry. It avoids scanning the flat condition
entries in `decisions.py`. The file has no learner dependency or JSON loading.
Its only import is a privately aliased standard-library `math` module.

Insert its contents after imports in `triton_unified_attention.py`, then call
its helper from host dispatch after extracting eligible, capture-compatible
features:

```python
config = _select_ua_tuned_config(device_name=device_name, **features)
if config is not None:
    # Use the complete launch settings from config.
    ...
# A None result requests the caller's existing heuristic fallback.
```

The helper and all its globals use UA-specific names so embedding does not
replace generic names in the host module. It preserves domain checks and config
legality, returns a copy of the chosen config, and records model/exporter hashes
in comments. The combined JSON policy described below offers an interpreted
alternative to embedding Python selectors.

An existing model can be exported without rerunning training or benchmarks:

```python
import json
from pathlib import Path
from ua_adapter import export_inline_python  # scripts/vllm/ua_tuning on sys.path

model = json.loads(Path("reports/ua-tree/tree.json").read_text())
Path("inline_selector.py").write_text(export_inline_python(model))
```

## Objective and search

Each usable workload/config pair uses the median of its saved timing samples.
Its normalized regret is `latency / fastest_valid_measured_latency - 1`.
The objective is weighted mean regret plus `leaf_penalty * leaves`. Weights are
positive `weight` fields in the workload records, defaulting to one. They are
normalized once across the training set, so child costs add correctly.
A penalty of `0.002` charges 0.2 percentage points of mean regret per leaf.
The objective controls average regret; it does not impose a worst-case regret cap.

Every leaf selects a complete config measured successfully for every workload
reaching it. Unknown and failed pairs cannot be selected. A config that never
wins for an individual workload remains eligible as a shared compromise.
Identical allowed feature vectors are grouped while preserving each workload's
weight and regret; incompatible measurements in an inseparable group are an error.
No failed workload is silently removed.

The learner starts with a complete greedy tree, then evaluates the depth-limited
DP recurrence over workload subsets. Memoization keys include the subset and
remaining depth. Branch-and-bound uses the sum of each inseparable group's best
shared cost plus the minimum leaf penalties. It never treats an unsplit child's
cost as a lower bound. Equivalent measured partitions are deduplicated.

`optimality_proven` means the search completed for the selected features,
observed predicates, depth, penalty, and measurements, within the reported
floating-point tolerance. It says nothing about unmeasured workloads or configs.
`--time-limit SECONDS` limits exact search after the initial greedy construction;
`--max-states COUNT` limits visited DP states. A limited run retains its best
complete tree and reports that optimality is unproven. Start with depth 3 and
increase it while examining regret, leaf count, runtime, and cached-state count.
The report also records peak process RSS on supported platforms; this includes
input loading and export validation, not just optimizer caches.

## Features and validity

The default learner features are the current key fields:

| Numerical comparisons | Categorical equality |
| --- | --- |
| `block_size`, `head_size`, `num_queries_per_kv`, `sliding_window` | `use_td`, `use_fp8_kv`, `is_3d` |
| `num_seqs`, `query_work` | `q_len_bucket` |
| | `q_element_size`, `k_element_size`, `v_element_size`, `out_element_size` |

Measurement records can include `grid_fill` and `avg_query_len`. They are
optional learner features, excluded from default training and exported domain
guards. To experiment with them, include their names alongside the desired key
features in `--features NAME ...`. The adapter exposes `AVAILABLE_FEATURES` as
`DEFAULT_FEATURES + OPTIONAL_FEATURES`; default training does not require the
optional fields when raw workload data is unavailable.

Use `--features NAME ...` to choose learned branches. Numerical thresholds are
midpoints between adjacent observed values. Categorical predicates test equality.
The default `--domain-policy supported-values` checks static/path/query key values
individually and permits new combinations of those supported values. Dynamic
numerical features (`num_seqs`, `query_work`, and optional `grid_fill` /
`avg_query_len`) accept any positive finite value. They are neither clamped nor
bounded by training minima/maxima; the actual values follow learned branches
into the end leaves. Static numerical support restrictions remain. An untrained
head size or GPU still uses fallback. `--domain-policy observed-regimes`
additionally requires the complete static-key/path/query-bucket tuple to have
appeared in training; use it
when exact training-regime coverage is desired. The policy is stored in the JSON
model, report, and generated selector. Older artifacts without an explicit policy
retain their original exact-regime behavior. Older domains that explicitly store
dynamic bounds in `numeric_ranges` retain those bounds until regenerated.
Segment count, raw KV lengths, KV buckets, softcap, and full dtype/quantization
tuples are not learned features. Segment count remains runtime launch metadata;
measurement producers can record it separately as `launch_metadata.num_segments`.
The adapter filters legacy key fields out of learner inputs, so existing saved
measurements can be reused without editing their original keys or timings. The
original full workloads and dtype metadata remain in the measurement artifacts.

The adapter checks complete configs against the current full-GQA search space:
`block_m` and `tile_size` in `{16, 32, 64, 128}`, warps in `{2, 4, 8, 16}`,
stages in `{1, 2, 3}`, and `grf_mode="256"`. It applies page divisibility, FP8 tile
restrictions, and Q/O tensor-descriptor constraints. A GQA ratio of 7 does not
require `block_m` to be divisible by 7; it requires at least one full GQA group.

The reusable engine is in `learner.py`. Other kernel adapters can use
`Dataset.from_records(records, feature_specs, valid_config)` with records shaped
as `{"id": ..., "features": {...}, "weight": 1, "options": [...]}`. Each option
contains a complete JSON `config`, `status`, and repeated `samples_ms`.

## Validation and deployment

```bash
python scripts/vllm/ua_tuning/learn.py path/to/measurements.json \
  --output-dir reports/ua-tree-holdout --max-depth 6 \
  --validation-fraction 0.2 --seed 1729
python -m pytest -s --tb=short scripts/vllm/ua_tuning/tests/test_learner.py
```

The holdout keeps identical allowed feature vectors together, including the two
KV allocation sizes in the current sweep. Training thresholds, costs, and domains
use only training rows. The report lists every held-out workload, including
unmeasured choices and out-of-domain fallbacks. Full-set regret is undefined when
any selection lacks a measured timing; covered-only metrics are labeled separately.
This is validation against saved measurements, not a new GPU benchmark.

All three Python exports are compared with the JSON tree interpreter on input
records, values at and immediately around numerical thresholds, and dynamic
values beyond training ranges, including 500 sequences and invalid inputs.
Tests also compare the exact optimizer against independent exhaustive enumeration,
including a case where DP improves on a greedy tree despite no immediate gain
from the first split.

The generated contract is:

```python
config = select_config(device_name=device_name, **features)
```

It returns a new complete config dictionary or `None` to request the existing
heuristic fallback. Missing/nonfinite features, nonpositive dynamic values,
unsupported key values, an unknown GPU, or a structurally invalid selected config
use fallback.
Under the default policy, unseen combinations of individually supported values
are evaluated by the tree. Exact tuple matching is optional.
The caller must retain the kernel's layout and quantization eligibility checks
and select an artifact for its recorded software environment. The emitted
Python selectors remain standalone; `policy.json` uses the standalone interpreter
described below.

Feature extraction belongs in host dispatch. Supply values consistent with graph
capture metadata: replay does not rerun Python decisions. In particular, actual
maximum query length can differ for batches sharing a captured graph;
`q_len_bucket` must follow the capture regime if used. Request counts can also
be less specific in piecewise graphs. The learner does not infer graph keys.

Correctness failures (`correct=False` or `correctness="failed"`) are excluded.
`--require-correctness` additionally requires an explicit passed result.
The current sweep has performance timings without per-candidate correctness
records, so its trees remain candidates for GPU correctness and integration
benchmarks. Held-out GPU workloads and new measurements inside threshold gaps
are required to assess deployment generalization. Existing-dispatcher speedups
require timing that dispatcher; regret relative to the measured oracle is not
such a speedup.

## Combined JSON policy interpreter

`learn.py` exports `policy.json` automatically. It contains one deduplicated pool
of complete configs, measured training anchors, the learned tree, the support
domain, and target/provenance information. Anchors and tree leaves reference the
same integer config IDs.

Selection uses an exact match on the policy's key fields first, then the tree on
a miss. Unsupported inputs or illegal configs return a fallback selection.
Positive finite dynamic values can exceed training bounds and follow the learned
end branches. The default domain permits new combinations of supported values.

Load and compile the policy once, then reuse it for subsequent selections:

```python
import json
from pathlib import Path
from policy_runtime import CompiledPolicy

artifact = json.loads(Path("reports/ua-tree/policy.json").read_text())
policy = CompiledPolicy(artifact, expected_target=current_target)
selection = policy.select(features, device_name=device_name)
if selection.config is not None:
    config = selection.config
else:
    # Apply the kernel's existing heuristic.
    ...
```

`current_target` describes the current device, software, and source identities
in the same format as the saved target. The caller supplies these independently
of the artifact and handles load/validation errors with its existing fallback.
The interpreter requires no learner imports. It validates and indexes the policy
at construction; the host can cache selections separately when appropriate.
Host integration owns file discovery, cache invalidation, launch eligibility,
and tuner override precedence.

Feature extraction should use host-visible shapes and capture metadata, with no
device content reads. Warm the ordinary attention path before graph capture.
Replay executes captured kernels and does not rerun Python selection.

Every exact anchor minimizes weighted mean normalized regret across all training
workloads with that key, restricted to configs measured successfully for every
one of them. This matches the learner's regret objective without its leaf penalty.
Multiple raw workloads can share a key, so the exporter never overwrites one row
with another. It requires precisely the saved training IDs and excludes validation
rows. The tree continues training on the entire training set.

`report.json` includes `combined_policy.training` and, when applicable,
`combined_policy.validation`, with per-workload selections, source counts,
measured coverage, and regret. An unmeasured selection makes full-set mean regret
undefined; it remains visible in the report. Seen-key improvements measure
reconstruction. Training/validation IDs stay in the tree/report, while runtime
anchors contain only keys and config references.

The standalone interpreter retains experimental joint-neighbor selection through
`on_miss={"method":"joint-nearest", "weights":{"num_seqs":1,"query_work":1},
"max_distance":null}`. It matches static/regime fields exactly and minimizes the
weighted sum of absolute log2 ratios over ranged features. An absent compatible
anchor or a distance beyond the optional radius uses the tree. Exported policies
default to exact-then-tree selection. Choose future distance settings on
training-only validation and evaluate fresh workloads before changing dispatch.
