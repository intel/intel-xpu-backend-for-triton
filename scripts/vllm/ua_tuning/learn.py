# SPDX-License-Identifier: Apache-2.0
"""Train a UA config decision tree from full offline sweep JSON measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from learner import Dataset, ExactLearner, canonical, evaluate, predict
from policy import build_policy, evaluate_policy
from ua_adapter import (
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
    adapt_records,
    domain_contains,
    export_decisions_json,
    export_decisions_python,
    export_inline_python,
    export_python,
    feature_specs,
    make_domain,
    select_model,
    valid_config,
)


def identity(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def metadata_context(data):
    records = data.get("per_workload", [{"metadata": data}])
    contexts = []
    for record in records:
        info = record["metadata"]
        contexts.append({
            "device_name": info["hardware"]["name"],
            "hardware": {name: info["hardware"].get(name)
                         for name in ("device_id", "driver_version")},
            "software": {name: info["software"].get(name)
                         for name in ("torch", "torch_git_revision", "triton")},
            "sources":
            {name: info["sources"][name]["sha256"]
             for name in ("kernel", "runtime_config", "triton_compiler")},
            "executed_tuner_sha256": info.get("executed_tuner_sha256"),
            "timing": {
                name: info["timing"].get(name)
                for name in (
                    "method",
                    "cache_policy",
                    "cache_eviction_bytes",
                    "round_statistic",
                    "latency_unit",
                    "warmup_ms",
                    "repetition_ms",
                    "rounds",
                )
            },
        })
        contexts[-1]["sources"].update(
            {name: info["sources"].get(name, {}).get("sha256")
             for name in ("timer", "ci_adapter", "ci_tuner_patch")})
    if not contexts or len({canonical(value) for value in contexts}) != 1:
        raise ValueError("Measurements mix different devices, software, kernel sources, or timing methods")
    return contexts[0]


def load_inputs(paths):
    rows, contexts, files = [], [], []
    for path in paths:
        path = Path(path)
        metadata = path.with_suffix(".metadata.json")
        if not metadata.is_file():
            raise ValueError(f"Missing adjacent benchmark metadata: {metadata}")
        content = json.loads(path.read_text())
        if not isinstance(content, list):
            raise TypeError(f"Expected a list of workload records: {path}")
        rows.extend(content)
        contexts.append(metadata_context(json.loads(metadata.read_text())))
        files.extend([identity(path), identity(metadata)])
        manifest = path.with_suffix(".manifest.json")
        if manifest.is_file():
            files.append(identity(manifest))
    if not contexts or len({canonical(value) for value in contexts}) != 1:
        raise ValueError("Input files belong to different benchmark environments")
    return adapt_records(rows), contexts[0], files


def split_records(records, specs, fraction, seed):
    if not 0 <= fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if fraction == 0:
        return records, []
    groups = {}
    for row in records:
        key = canonical([row["features"][feature.name] for feature in specs])
        groups.setdefault(key, []).append(row)
    if len(groups) < 2:
        raise ValueError("Need at least two distinct feature vectors for held-out validation")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = min(len(keys) - 1, max(1, round(fraction * len(keys))))
    held_out = set(keys[:count])
    train, validation = [], []
    for key, group in groups.items():
        (validation if key in held_out else train).extend(group)
    return train, validation


def check_export(model, records):
    namespace = {}
    code = export_python(model)
    exec(compile(code, "<generated UA selector>", "exec"), namespace)  # noqa: S102 - verify our generated selector
    generated = namespace["select_config"]
    dictionary_namespace = {}
    dictionary_code = export_decisions_python(model)
    exec(compile(dictionary_code, "<generated UA dictionary>", "exec"), dictionary_namespace)  # noqa: S102
    dictionary_selector = dictionary_namespace["select_config"]
    inline_namespace = {}
    inline_code = export_inline_python(model)
    exec(compile(inline_code, "<generated inline UA selector>", "exec"), inline_namespace)  # noqa: S102
    selectors = (generated, dictionary_selector, inline_namespace["_select_ua_tuned_config"])
    device = model["target"]["device_name"]
    checks = 0
    feature_rows = [dict(row["features"]) for row in records]

    def boundaries(node):
        if "config" in node:
            return
        if node["operator"] == "<=":
            threshold = node["value"]
            for features in [row["features"] for row in records]:
                for value in (math.nextafter(threshold, -math.inf), threshold, math.nextafter(threshold, math.inf)):
                    feature_rows.append(dict(features, **{node["feature"]: value}))
        boundaries(node["left"])
        boundaries(node["right"])

    boundaries(model["tree"])
    for name in model["domain"].get("positive_numeric_features", []):
        largest = max(row["features"][name] for row in records)
        for row in records:
            for value in (1, 500, max(10**12, largest * 2), 0, -1, math.nan, math.inf):
                feature_rows.append(dict(row["features"], **{name: value}))
    for features in feature_rows:
        expected = select_model(model, features, device)
        if any(selector(device_name=device, **features) != expected for selector in selectors):
            raise ValueError(f"Export/tree disagreement: {features}")
        checks += 1
    for selector in selectors:
        if selector(device_name="unseen GPU", **records[0]["features"]) is not None:
            raise ValueError("Generated selector did not fall back for an unseen GPU")
    for spec in model["features"]:
        features = dict(records[0]["features"])
        del features[spec["name"]]
        for selector in selectors:
            if selector(device_name=device, **features) is not None:
                raise ValueError(f"Generated selector did not guard missing {spec['name']}")
        checks += 1
    return checks


def guarded_report(tree, dataset, examples, model, penalty):
    result = evaluate(tree, dataset, examples, penalty)
    guarded = 0
    for row, example in zip(result["per_workload"], examples):
        if not domain_contains(model["domain"], example.features):
            row.update(status="fallback_out_of_domain", latency_ms=None, regret=None)
            guarded += 1
        elif not valid_config(dataset.configs[predict(tree, example.features)], example.features):
            row.update(status="fallback_invalid_config", latency_ms=None, regret=None)
            guarded += 1
    result["fallbacks"] = guarded
    if guarded:
        values = [(row["weight"], row["regret"]) for row in result["per_workload"] if row["regret"] is not None]
        result.update(
            measured=len(values),
            coverage=len(values) / len(examples),
            weighted_mean_regret=None,
            objective=None,
            covered_weighted_mean_regret=(math.fsum(w * v for w, v in values) /
                                          math.fsum(w for w, _ in values) if values else None),
            max_regret=max((v for _, v in values), default=None),
        )
        regrets = sorted(v for _, v in values)
        result["p95_regret"] = regrets[math.ceil(0.95 * len(regrets)) - 1] if regrets else None
    return result


def train(args):
    records, context, inputs = load_inputs(args.measurements)
    specs = feature_specs(args.features)
    # Validate all rows before splitting; test data never contributes thresholds or costs.
    all_data = Dataset.from_records(records, specs, valid_config, args.require_correctness)
    training, validation = split_records(records, specs, args.validation_fraction, args.seed)
    dataset = Dataset.from_records(training, specs, valid_config, args.require_correctness)
    learner = ExactLearner(dataset, args.max_depth, args.leaf_penalty, args.time_limit, args.max_states)
    print(
        f"Training {len(training)} workloads ({len(dataset.groups)} distinct feature vectors), "
        f"{len(dataset.configs)} configs, {len(dataset.predicates)} predicates",
        flush=True,
    )
    tree, greedy, stats = learner.fit()
    model = {
        "schema_version": 1,
        "adapter": "ua_full_gqa_v1",
        "target": context,
        "features": [{"name": feature.name, "kind": feature.kind} for feature in specs],
        "configs": list(dataset.configs),
        "tree": tree.to_dict(),
        "domain": make_domain(training, specs, policy=args.domain_policy),
        "objective": {
            "name": "weighted_mean_normalized_regret_plus_leaf_penalty",
            "leaf_penalty": args.leaf_penalty,
            "latency_statistic": "median_samples_ms",
            "weights": "record_weight_or_one",
        },
        "settings": {
            "domain_policy": args.domain_policy,
            "require_correctness": args.require_correctness,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        },
        "search": dict(stats, max_depth=args.max_depth, time_limit=args.time_limit, max_states=args.max_states),
        "training_ids": [row["id"] for row in training],
        "validation_ids": [row["id"] for row in validation],
        "provenance": {
            "created_at_utc":
            datetime.now(timezone.utc).isoformat(),
            "inputs":
            inputs,
            "learner_sources": [
                identity(Path(__file__).with_name(name))
                for name in ("learner.py", "ua_adapter.py", "learn.py", "policy.py", "policy_runtime.py")
            ],
        },
    }
    policy = build_policy(model, training)
    checks = check_export(model, records)
    report = {
        "schema_version":
        1,
        "search":
        model["search"],
        "export_parity_checks":
        checks,
        "dictionary_export_parity_checks":
        checks,
        "inline_export_parity_checks":
        checks,
        "training":
        guarded_report(tree, dataset, dataset.examples, model, args.leaf_penalty),
        "combined_policy": {
            "anchors": len(policy["anchors"]),
            "configs": len(policy["configs"]),
            "training": evaluate_policy(policy, dataset.examples),
        },
        "greedy":
        evaluate(greedy, dataset, leaf_penalty=args.leaf_penalty),
        "measurement_status_counts":
        dict(Counter(option.get("status", "unknown") for row in records for option in row.get("options", []))),
        "correctness_evidence":
        dict(
            Counter(
                str(option.get("correctness", option.get("correct", "not_recorded")))
                for row in records
                for option in row.get("options", []))),
        "validation_policy": {
            "fraction": args.validation_fraction,
            "seed": args.seed,
            "group": "identical allowed feature vectors stay together",
            "meaning": "offline timing-matrix holdout; no new GPU measurements",
        },
        "deployment": {
            "domain_policy": args.domain_policy,
            "contract": "select_config(*, device_name, **features) -> complete config dict or None",
            "fallback": "caller uses its existing heuristic when None is returned",
            "integration": "UA loads policy.json through VLLM_UA_POLICY_FILE or VLLM_TUNED_CONFIG_FOLDER",
            "selection": "exact measured key, otherwise learned tree, otherwise existing heuristic",
            "graph_capture": "supply capture-compatible host metadata; replay does not rerun Python selection",
            "numeric_extrapolation": "positive finite dynamic values use actual values beyond training ranges",
        },
    }
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["resources"] = {"process_peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024}
    except ImportError:
        report["resources"] = {"process_peak_rss_bytes": None}
    leaf = learner.leaf(dataset.full_mask)
    report["single_config"] = evaluate(leaf.tree, dataset, leaf_penalty=args.leaf_penalty) if leaf else None
    # A lower bound: every indistinguishable feature group gets its best shared config.
    report["feature_oracle_mean_regret"] = math.fsum(dataset.atom_minima)
    if validation:
        examples = {example.identifier: example for example in all_data.examples}
        report["validation"] = guarded_report(tree, dataset, [examples[row["id"]] for row in validation], model,
                                              args.leaf_penalty)
        report["combined_policy"]["validation"] = evaluate_policy(policy, [examples[row["id"]] for row in validation])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("tree.json", model),
        ("policy.json", policy),
        ("report.json", report),
        ("decisions.json", export_decisions_json(model)),
    ):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (args.output_dir / "selector.py").write_text(export_python(model))
    (args.output_dir / "decisions.py").write_text(export_decisions_python(model))
    (args.output_dir / "inline_selector.py").write_text(export_inline_python(model))
    print(
        json.dumps({
            "output_dir": str(args.output_dir.resolve()),
            "optimality_proven": stats["optimality_proven"],
            "stopping_reason": stats["stopping_reason"],
            "leaves": tree.leaves,
            "depth": tree.depth,
            "mean_regret": report["training"]["weighted_mean_regret"],
            "max_regret": report["training"]["max_regret"],
            "objective": stats["objective"],
            "elapsed_seconds": stats["elapsed_seconds"],
            "export_parity_checks": checks,
        }),
        flush=True,
    )
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--leaf-penalty", type=float, default=0.002,
                        help="Penalty per leaf in normalized-regret units (default: 0.002)")
    parser.add_argument(
        "--time-limit",
        type=float,
        help="Exact-search seconds after the initial feasible greedy tree; retains incumbent on timeout",
    )
    parser.add_argument("--max-states", type=int, help="Maximum visited DP states; retains incumbent on limit")
    parser.add_argument("--features", nargs="+", default=list(DEFAULT_FEATURES), choices=AVAILABLE_FEATURES)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--domain-policy",
        choices=("supported-values", "observed-regimes"),
        default="supported-values",
        help="Allow new combinations of supported feature values (default), or require an exact observed regime",
    )
    parser.add_argument("--require-correctness", action="store_true",
                        help="Use only options with explicit passed correctness evidence")
    args = parser.parse_args(argv)
    try:
        train(args)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
