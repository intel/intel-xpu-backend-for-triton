# SPDX-License-Identifier: Apache-2.0
"""Export a shared config pool, measured exact anchors, and a learned UA tree."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from learner import Dataset, canonical
from policy_runtime import CompiledPolicy, Selection  # noqa: F401 - public selection result
from ua_adapter import DEFAULT_FEATURES, feature_specs, valid_config


def build_policy(model, training_records):
    """Export all training anchors and tree leaves into one compact config pool."""
    if model.get("adapter") != "ua_full_gqa_v1" or model.get("schema_version") != 1:
        raise ValueError("Unsupported source model")
    ids = [row["id"] for row in training_records]
    if len(set(ids)) != len(ids) or set(ids) != set(model["training_ids"]):
        raise ValueError("Anchor records must match the model training IDs exactly")
    if set(ids) & set(model.get("validation_ids", [])):
        raise ValueError("Validation rows cannot become training anchors")
    names = tuple(dict.fromkeys(DEFAULT_FEATURES + tuple(f["name"] for f in model["features"])))
    data = Dataset.from_records(training_records, feature_specs(names), valid_config,
                                model["settings"].get("require_correctness", False))
    winners = [min(range(len(data.configs)), key=lambda c: losses[c]) for losses in data.losses]
    used = {canonical(data.configs[c]) for c in winners}

    def collect(node):
        if "config" in node:
            used.add(canonical(model["configs"][node["config"]]))
        else:
            collect(node["left"])
            collect(node["right"])

    collect(model["tree"])
    configs = [json.loads(value) for value in sorted(used)]
    config_ids = {canonical(config): index for index, config in enumerate(configs)}

    def remap(node):
        if "config" in node:
            return {"config": config_ids[canonical(model["configs"][node["config"]])]}
        return dict(node, left=remap(node["left"]), right=remap(node["right"]))

    artifact = {
        "schema_version":
        1,
        "adapter":
        "ua_json_policy_v1",
        "target":
        model["target"],
        "key_fields":
        list(names),
        "features":
        model["features"],
        "domain":
        model["domain"],
        "configs":
        configs,
        "anchors": [{
            "key": [features[name] for name in names],
            "config": config_ids[canonical(data.configs[winner])],
        } for features, winner in zip(data.atom_features, winners)],
        "tree":
        remap(model["tree"]),
        "on_miss": {"method": "tree"},
        "objective": {
            "anchor": "weighted_mean_normalized_regret",
            "latency_statistic": "median_samples_ms",
            "shared_successful_configs_only": True,
            "tree": model["objective"],
        },
        "provenance": {
            "model_sha256": hashlib.sha256(canonical(model).encode()).hexdigest(),
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interpreter_sha256":
            hashlib.sha256(Path(__file__).with_name("policy_runtime.py").read_bytes()).hexdigest(),
            "inputs": model["provenance"]["inputs"],
        },
    }
    # A JSON round trip detaches the artifact from mutable training/model objects.
    artifact = json.loads(canonical(artifact))
    CompiledPolicy(artifact, expected_target=model["target"])
    return artifact


def evaluate_policy(artifact, examples):
    """Report combined selections against saved timings, including unmeasured choices."""
    policy = CompiledPolicy(artifact, expected_target=artifact["target"])
    rows, values, sources = [], [], Counter()
    total_weight = math.fsum(example.weight for example in examples)
    for example in examples:
        result = policy.select(example.features, device_name=artifact["target"]["device_name"])
        sources[result.source] += 1
        latency = example.latencies.get(canonical(result.config)) if result.config is not None else None
        regret = latency / min(example.latencies.values()) - 1 if latency is not None else None
        if regret is not None:
            values.append((example.weight, regret))
        rows.append({
            "id": example.identifier,
            "source": result.source,
            "config": result.config,
            "status": "fallback" if result.config is None else "measured" if latency is not None else "unmeasured",
            "latency_ms": latency,
            "regret": regret,
            "weight": example.weight,
        })
    covered_weight = math.fsum(weight for weight, _ in values)
    cost = math.fsum(weight * regret for weight, regret in values)
    ordered = sorted(regret for _, regret in values)
    return {
        "workloads": len(rows),
        "measured": len(values),
        "coverage": len(values) / len(rows),
        "source_counts": dict(sources),
        "weighted_mean_regret": cost / total_weight if len(values) == len(rows) else None,
        "covered_weighted_mean_regret": cost / covered_weight if covered_weight else None,
        "max_regret": max(ordered, default=None),
        "p95_regret": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
        "per_workload": rows,
    }
