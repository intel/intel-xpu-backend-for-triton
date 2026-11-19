# SPDX-License-Identifier: Apache-2.0
"""Shared standard-library interpreter for UA exact-anchor/tree JSON policies.

Host dispatch can reuse this interpreter without importing the learner.
The caller supplies current target identities, eligibility checks, and fallback.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

STATIC_FIELDS = (
    "block_size",
    "head_size",
    "num_queries_per_kv",
    "use_td",
    "use_fp8_kv",
    "sliding_window",
    "q_element_size",
    "k_element_size",
    "v_element_size",
    "out_element_size",
)
REGIME_FIELDS = STATIC_FIELDS + ("is_3d", "q_len_bucket")
DEFAULT_FEATURES = REGIME_FIELDS + ("num_seqs", "query_work")
OPTIONAL_FEATURES = ("grid_fill", "avg_query_len")
DYNAMIC_NUMERIC_FEATURES = ("num_seqs", "query_work") + OPTIONAL_FEATURES
AVAILABLE_FEATURES = DEFAULT_FEATURES + OPTIONAL_FEATURES
CATEGORICAL = {
    "use_td",
    "use_fp8_kv",
    "is_3d",
    "q_len_bucket",
    "q_element_size",
    "k_element_size",
    "v_element_size",
    "out_element_size",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def power_of_two(value):
    return type(value) is int and value > 0 and value & (value - 1) == 0


def valid_config(config, features):
    """Mirror the branch's structural constraints and fixed full-GQA search space."""
    if not isinstance(config, dict) or set(config) != {"block_m", "tile_size", "num_warps", "num_stages", "grf_mode"}:
        return False
    group, page, head = (features[name] for name in ("num_queries_per_kv", "block_size", "head_size"))
    m, tile, warps, stages = (config[name] for name in ("block_m", "tile_size", "num_warps", "num_stages"))
    if any(type(value) is not int for value in (group, page, head, m, tile, warps, stages)):
        return False
    if group <= 0 or page <= 0 or head <= 0 or m not in (16, 32, 64, 128) or m < group:
        return False
    if tile not in (16, 32, 64, 128) or warps not in (2, 4, 8, 16) or stages not in (1, 2, 3):
        return False
    if config["grf_mode"] != "256" or page < tile or page % tile:
        return False
    if features["q_element_size"] == 1 and tile < 32:
        return False
    if features["use_td"] and power_of_two(head) and power_of_two(group):
        return power_of_two(m // group)
    return True


def domain_contains(domain, features):
    """Check supported keys and valid numbers; dynamic values may exceed training bounds."""
    try:
        # Older artifacts without a policy retain their original exact-tuple guard.
        policy = domain.get("policy", "observed-regimes")
        if policy not in ("supported-values", "observed-regimes"):
            return False
        if policy == "supported-values" and not domain.get("supported_values"):
            return False
        for name, values in domain.get("supported_values", {}).items():
            if features[name] not in values:
                return False
        if policy == "observed-regimes":
            regime = [features[name] for name in domain["regime_fields"]]
            if regime not in domain["regimes"]:
                return False
        for name in domain.get("positive_numeric_features", []):
            value = features[name]
            if not number(value) or value <= 0:
                return False
        for name, (low, high) in domain["numeric_ranges"].items():
            value = features[name]
            if not number(value) or not low <= value <= high:
                return False
        for name, values in domain["categorical_values"].items():
            if features[name] not in values:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _validate_names(names):
    if not names or len(set(names)) != len(names) or set(names) - set(AVAILABLE_FEATURES):
        raise ValueError("Invalid or duplicate policy feature names")


@dataclass(frozen=True)
class Selection:
    config: dict | None
    source: str
    distance: float | None = None
    anchor: tuple | None = None


class CompiledPolicy:
    """Validate once and index anchors; callers can cache this object by target."""

    @classmethod
    def load(cls, path, *, expected_target):
        return cls(json.loads(Path(path).read_text()), expected_target=expected_target)

    def __init__(self, artifact, *, expected_target):
        self.data = json.loads(canonical(artifact))
        data = self.data
        if data.get("schema_version") != 1 or data.get("adapter") != "ua_json_policy_v1":
            raise ValueError("Unsupported JSON policy schema/adapter")
        if data["target"] != expected_target:
            raise ValueError("Policy target does not match the requested benchmark environment")
        self.key_fields = tuple(data["key_fields"])
        _validate_names(self.key_fields)
        if not set(DEFAULT_FEATURES) <= set(self.key_fields):
            raise ValueError("Policy must retain every UA key field")
        names = tuple(feature["name"] for feature in data["features"])
        _validate_names(names)
        if not set(names) <= set(self.key_fields):
            raise ValueError("Tree features must belong to the policy key")
        self.range_fields = tuple(name for name in self.key_fields if name in DYNAMIC_NUMERIC_FEATURES)
        self.configs = data["configs"]
        if not self.configs or len({canonical(c) for c in self.configs}) != len(self.configs):
            raise ValueError("Policy configs must be nonempty and distinct")
        kinds = {name: "categorical" if name in CATEGORICAL else "numeric" for name in names}
        if any(feature["kind"] != kinds[feature["name"]] for feature in data["features"]):
            raise ValueError("Invalid policy feature kind")

        def check_index(index):
            if type(index) is not int or not 0 <= index < len(self.configs):
                raise ValueError("Invalid policy config reference")

        def check_tree(node, depth=0):
            if not isinstance(node, dict) or depth > 64:
                raise ValueError("Malformed or excessively deep policy tree")
            if set(node) == {"config"}:
                check_index(node["config"])
                return
            if set(node) != {"feature", "operator", "value", "left", "right"}:
                raise ValueError("Malformed policy tree")
            kind = kinds.get(node["feature"])
            if kind is None or node["operator"] != ("<=" if kind == "numeric" else "=="):
                raise ValueError("Invalid policy tree predicate")
            if kind == "numeric" and not number(node["value"]):
                raise ValueError("Invalid policy threshold")
            check_tree(node["left"], depth + 1)
            check_tree(node["right"], depth + 1)

        check_tree(data["tree"])
        self.tree = data["tree"]
        self.exact = {}
        self.partitions = defaultdict(list)
        for anchor in data["anchors"]:
            key = tuple(anchor["key"])
            check_index(anchor["config"])
            if len(key) != len(self.key_fields) or key in self.exact:
                raise ValueError("Invalid or duplicate policy anchor")
            features = dict(zip(self.key_fields, key))
            if not self._supported(features) or not valid_config(self.configs[anchor["config"]], features):
                raise ValueError("Unsupported or illegal policy anchor")
            self.exact[key] = anchor["config"]
            self.partitions[tuple(features[name] for name in REGIME_FIELDS)].append((key, anchor["config"]))
        on_miss = data["on_miss"]
        if on_miss["method"] not in ("tree", "joint-nearest"):
            raise ValueError("Unknown on-miss method")
        self.weights = on_miss.get("weights", {name: 1 for name in self.range_fields})
        self.radius = on_miss.get("max_distance")
        if set(self.weights) != set(self.range_fields) or any(not number(value) or value <= 0
                                                              for value in self.weights.values()):
            raise ValueError("Distance weights must be positive and cover the ranged fields")
        if self.radius is not None and (not number(self.radius) or self.radius < 0):
            raise ValueError("Maximum distance must be finite and nonnegative")

    def _supported(self, features):
        if not domain_contains(self.data["domain"], features):
            return False
        # The anchor key stays complete even when the learner uses fewer features.
        return all(number(features.get(name)) and features[name] > 0 for name in self.range_fields)

    def select(self, features, *, device_name):
        if device_name != self.data["target"]["device_name"] or not self._supported(features):
            return Selection(None, "fallback")
        try:
            key = tuple(features[name] for name in self.key_fields)
            index = self.exact.get(key)
            source, distance, anchor_key = "exact", None, None
            if index is None and self.data["on_miss"]["method"] == "joint-nearest":
                partition = self.partitions.get(tuple(features[name] for name in REGIME_FIELDS), ())

                def distance_to(entry):
                    neighbor = dict(zip(self.key_fields, entry[0]))
                    return math.fsum(self.weights[name] * abs(math.log2(neighbor[name]) - math.log2(features[name]))
                                     for name in self.range_fields)

                if partition:
                    entry = min(partition, key=lambda item: (distance_to(item), item[0], item[1]))
                    distance = distance_to(entry)
                    if self.radius is None or distance <= self.radius:
                        anchor_key, index = entry
                        source = "joint-nearest"
            if index is None:
                node = self.tree
                while "config" not in node:
                    value = features[node["feature"]]
                    matches = value <= node["value"] if node["operator"] == "<=" else value == node["value"]
                    node = node["left"] if matches else node["right"]
                index, source, distance = node["config"], "tree", None
            config = self.configs[index]
            if not valid_config(config, features):
                return Selection(None, "fallback")
            return Selection(dict(config), source, distance, anchor_key)
        except (KeyError, TypeError, ValueError):
            return Selection(None, "fallback")
