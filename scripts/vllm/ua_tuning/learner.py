# SPDX-License-Identifier: Apache-2.0
"""Exact cost-sensitive trees for complete offline configuration measurements.

The engine uses only the standard library. Kernel adapters supply feature records,
configuration validity, and deployment guards. Costs are normalized once against
all training weights, so child costs add without changing the objective.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str = "numeric"

    def __post_init__(self):
        if not self.name or self.kind not in ("numeric", "categorical"):
            raise ValueError(f"Invalid feature specification: {self}")


@dataclass
class Example:
    identifier: str
    features: dict
    latencies: dict[str, float]
    weight: float = 1.0


@dataclass(frozen=True)
class Predicate:
    feature: str
    operator: str
    value: object

    def matches(self, features):
        value = features[self.feature]
        return value <= self.value if self.operator == "<=" else value == self.value


@dataclass(frozen=True)
class Tree:
    config: int | None = None
    predicate: Predicate | None = None
    left: Tree | None = None
    right: Tree | None = None

    @property
    def leaves(self):
        return 1 if self.predicate is None else self.left.leaves + self.right.leaves

    @property
    def depth(self):
        return 0 if self.predicate is None else 1 + max(self.left.depth, self.right.depth)

    def to_dict(self):
        if self.predicate is None:
            return {"config": self.config}
        return {
            "feature": self.predicate.feature,
            "operator": self.predicate.operator,
            "value": self.predicate.value,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, data):
        if "config" in data:
            return cls(config=data["config"])
        return cls(
            predicate=Predicate(data["feature"], data["operator"], data["value"]),
            left=cls.from_dict(data["left"]),
            right=cls.from_dict(data["right"]),
        )


def predict(tree, features):
    while tree.predicate is not None:
        tree = tree.left if tree.predicate.matches(features) else tree.right
    return tree.config


class Dataset:
    """A timing matrix; missing/failed combinations have infinite selection cost."""

    def __init__(self, examples, configs, features):
        self.examples = tuple(examples)
        self.features = tuple(features)
        self.configs = tuple(sorted(configs, key=canonical))
        self.config_ids = tuple(map(canonical, self.configs))
        if not self.examples or not self.configs or len(set(self.config_ids)) != len(self.config_ids):
            raise ValueError("Need nonempty examples and distinct complete configurations")
        if len({feature.name for feature in self.features}) != len(self.features):
            raise ValueError("Duplicate feature names")
        if len({example.identifier for example in self.examples}) != len(self.examples):
            raise ValueError("Duplicate workload IDs; merge repeated samples explicitly")
        for example in self.examples:
            if not number(example.weight) or example.weight <= 0:
                raise ValueError(f"{example.identifier}: weight must be finite and positive")
            if not example.latencies or any(not number(t) or t <= 0 for t in example.latencies.values()):
                raise ValueError(f"{example.identifier}: no valid positive timings")
            if set(example.latencies) - set(self.config_ids):
                raise ValueError(f"{example.identifier}: unknown configuration ID")
            for feature in self.features:
                value = example.features.get(feature.name)
                if feature.kind == "numeric":
                    valid = number(value)
                else:
                    valid = type(value) in (bool, int, float, str) and (not isinstance(value, float)
                                                                        or math.isfinite(value))
                if not valid:
                    raise ValueError(f"{example.identifier}: invalid/missing feature {feature.name}")
        total_weight = math.fsum(example.weight for example in self.examples)
        groups = {}
        for index, example in enumerate(self.examples):
            key = tuple(example.features[feature.name] for feature in self.features)
            groups.setdefault(key, []).append(index)
        self.groups = tuple(groups[key] for key in sorted(groups, key=canonical))
        self.atom_features = tuple(self.examples[group[0]].features for group in self.groups)
        self.losses, self.valid_masks = [], []
        for group in self.groups:
            losses = []
            for identifier in self.config_ids:
                terms = []
                for index in group:
                    example = self.examples[index]
                    latency = example.latencies.get(identifier)
                    terms.append(math.inf if latency is None else example.weight / total_weight *
                                 (latency / min(example.latencies.values()) - 1))
                losses.append(math.fsum(terms))
            if not any(math.isfinite(loss) for loss in losses):
                identifiers = [self.examples[index].identifier for index in group]
                raise ValueError(f"Indistinguishable workloads have no shared measured valid config: {identifiers}")
            self.losses.append(tuple(losses))
            self.valid_masks.append(sum(1 << c for c, loss in enumerate(losses) if math.isfinite(loss)))
        self.atom_minima = tuple(map(min, self.losses))
        self.full_mask = (1 << len(self.groups)) - 1
        self.predicates = self._predicates()

    @classmethod
    def from_records(cls, records, features, valid_config: Callable | None = None, require_correctness=False):
        examples, configurations = [], {}
        for row in records:
            latencies = {}
            for option in row.get("options", []):
                config = option["config"]
                identifier = canonical(config)
                configurations[identifier] = config
                if identifier in latencies:
                    raise ValueError(f"{row['id']}: duplicate successful configuration")
                correctness = option.get("correctness")
                if (option.get("status") != "ok" or option.get("correct") is False
                        or correctness in ("failed", "error", False)):
                    continue
                if require_correctness and option.get("correct") is not True and correctness not in ("passed", True):
                    continue
                if valid_config is not None and not valid_config(config, row["features"]):
                    raise ValueError(f"{row['id']}: measured config violates adapter validity: {config}")
                samples = option.get("samples_ms", [])
                if not samples or any(not number(value) or value <= 0 for value in samples):
                    raise ValueError(f"{row['id']}: successful config has invalid timing samples")
                latencies[identifier] = statistics.median(samples)
            if not latencies:
                raise ValueError(f"{row['id']}: no usable measured config (unknown/failed pairs cannot be selected)")
            examples.append(Example(str(row["id"]), dict(row["features"]), latencies, row.get("weight", 1.0)))
        return cls(examples, configurations.values(), features)

    def _predicates(self):
        result, seen = [], set()
        for feature in self.features:
            values = sorted(
                {row[feature.name]
                 for row in self.atom_features},
                key=canonical if feature.kind == "categorical" else None,
            )
            if feature.kind == "numeric":
                choices = []
                for first, second in pairwise(values):
                    midpoint = first / 2 + second / 2
                    # Adjacent representable floats may have no representable midpoint.
                    choices.append(("<=", midpoint if first <= midpoint < second else first))
            else:
                choices = [("==", value) for value in values]
            for operator, value in choices:
                predicate = Predicate(feature.name, operator, value)
                left = sum(1 << index for index, row in enumerate(self.atom_features) if predicate.matches(row))
                right = self.full_mask ^ left
                signature = min(left, right)
                if left and right and signature not in seen:
                    seen.add(signature)
                    result.append((predicate, left))
        return tuple(result)


def indices(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


@dataclass(frozen=True)
class Solution:
    score: float
    tree: Tree


class SearchLimit(Exception):
    pass


class ExactLearner:
    """Memoized depth-limited DP with admissible branch-and-bound pruning.

    Bounds grant each indistinguishable feature group its best shared config.
    Each unfinished nonempty child pays at least one leaf penalty (two when it
    has no shared config). An unsplit child's measured cost is never a bound.
    """

    tolerance = 1e-12

    def __init__(self, dataset, max_depth=3, leaf_penalty=0.002, time_limit=None, max_states=None):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if not number(leaf_penalty) or leaf_penalty < 0:
            raise ValueError("leaf_penalty must be finite and nonnegative")
        if time_limit is not None and (not number(time_limit) or time_limit < 0):
            raise ValueError("time_limit must be finite and nonnegative")
        if max_states is not None and (type(max_states) is not int or max_states < 0):
            raise ValueError("max_states must be a nonnegative integer")
        self.data, self.max_depth, self.penalty = dataset, max_depth, leaf_penalty
        self.time_limit, self.max_states = time_limit, max_states
        self.leaf_cache, self.greedy_cache, self.cache, self.bound_cache = {}, {}, {}, {}
        self.stats = {"states": 0, "cache_hits": 0, "pruned": 0}
        self.deadline = None
        self.incumbent = None

    def leaf(self, mask):
        if mask not in self.leaf_cache:
            members = tuple(indices(mask))
            common = (1 << len(self.data.configs)) - 1
            for index in members:
                common &= self.data.valid_masks[index]
            if not common:
                self.leaf_cache[mask] = None
            else:
                loss, config = min(
                    (math.fsum(self.data.losses[index][c] for index in members), c) for c in indices(common))
                self.leaf_cache[mask] = Solution(loss + self.penalty, Tree(config=config))
        return self.leaf_cache[mask]

    def splits(self, mask):
        seen = set()
        for predicate, global_left in self.data.predicates:
            left = mask & global_left
            right = mask ^ left
            signature = min(left, right)
            if left and right and signature not in seen:
                seen.add(signature)
                yield predicate, left, right

    def lower_bound(self, mask, depth):
        key = mask, depth
        if key not in self.bound_cache:
            leaf = self.leaf(mask)
            if depth == 0:
                value = leaf.score if leaf else math.inf
            else:
                value = math.fsum(self.data.atom_minima[index] for index in indices(mask))
                value += self.penalty * (1 if leaf else 2)
            self.bound_cache[key] = value
        return self.bound_cache[key]

    @staticmethod
    def join(predicate, left, right):
        return Solution(left.score + right.score, Tree(predicate=predicate, left=left.tree, right=right.tree))

    def better(self, candidate, incumbent):
        return candidate is not None and (incumbent is None or candidate.score < incumbent.score - self.tolerance)

    def greedy(self, mask, depth):
        key = mask, depth
        if key in self.greedy_cache:
            return self.greedy_cache[key]
        best = self.leaf(mask)
        if depth:
            candidates = []
            for order, (predicate, left, right) in enumerate(self.splits(mask)):
                a, b = self.leaf(left), self.leaf(right)
                immediate = a.score + b.score if a and b else math.inf
                candidates.append((immediate, order, predicate, left, right))
            for immediate, _, predicate, left, right in sorted(candidates, key=lambda value: value[:2]):
                if best is not None and immediate >= best.score - self.tolerance:
                    break
                a, b = self.greedy(left, depth - 1), self.greedy(right, depth - 1)
                if a is not None and b is not None:
                    candidate = self.join(predicate, a, b)
                    if self.better(candidate, best):
                        best = candidate
                    # Greedy commits to its best immediate feasible split.
                    break
        self.greedy_cache[key] = best
        return best

    def check_limit(self):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise SearchLimit("time_limit")
        if self.max_states is not None and self.stats["states"] >= self.max_states:
            raise SearchLimit("state_limit")

    def solve(self, mask, depth):
        self.check_limit()
        key = mask, depth
        if key in self.cache:
            self.stats["cache_hits"] += 1
            return self.cache[key]
        self.stats["states"] += 1
        best = self.greedy(mask, depth)
        candidates = []
        if depth:
            for order, (predicate, left, right) in enumerate(self.splits(mask)):
                bound = self.lower_bound(left, depth - 1) + self.lower_bound(right, depth - 1)
                candidates.append((bound, order, predicate, left, right))
            for bound, _, predicate, left, right in sorted(candidates, key=lambda value: value[:2]):
                self.check_limit()
                if best is not None and bound > best.score + self.tolerance:
                    self.stats["pruned"] += 1
                    continue
                a = self.solve(left, depth - 1)
                if a is None:
                    continue
                if best is not None and a.score + self.lower_bound(right, depth - 1) > best.score + self.tolerance:
                    self.stats["pruned"] += 1
                    continue
                b = self.solve(right, depth - 1)
                if b is not None:
                    candidate = self.join(predicate, a, b)
                    if self.better(candidate, best):
                        best = candidate
                        if mask == self.data.full_mask and depth == self.max_depth:
                            self.incumbent = best
        self.cache[key] = best
        return best

    def fit(self):
        started = time.monotonic()
        self.incumbent = self.greedy(self.data.full_mask, self.max_depth)
        if self.incumbent is None:
            raise ValueError("No feasible tree at this depth using the allowed predicates and measured configs")
        greedy = self.incumbent
        self.stats["greedy_seconds"] = time.monotonic() - started
        self.deadline = None if self.time_limit is None else time.monotonic() + self.time_limit
        reason = "complete"
        try:
            self.incumbent = self.solve(self.data.full_mask, self.max_depth)
        except SearchLimit as error:
            reason = str(error)
        self.stats.update(
            elapsed_seconds=time.monotonic() - started,
            cached_states=len(self.cache),
            leaf_states=len(self.leaf_cache),
            atoms=len(self.data.groups),
            predicates=len(self.data.predicates),
            optimality_proven=reason == "complete",
            stopping_reason=reason,
            numerical_tolerance=self.tolerance,
            objective=self.incumbent.score,
            greedy_objective=greedy.score,
        )
        return self.incumbent.tree, greedy.tree, dict(self.stats)


def evaluate(tree, dataset, examples=None, leaf_penalty=0.0):
    examples = dataset.examples if examples is None else tuple(examples)
    rows, weighted = [], []
    for example in examples:
        config = predict(tree, example.features)
        latency = example.latencies.get(dataset.config_ids[config])
        best = min(example.latencies.values())
        regret = None if latency is None else latency / best - 1
        rows.append({
            "id": example.identifier,
            "config": config,
            "weight": example.weight,
            "status": "unmeasured_or_invalid" if latency is None else "measured",
            "latency_ms": latency,
            "oracle_ms": best,
            "regret": regret,
        })
        if regret is not None:
            weighted.append((example.weight, regret))
    complete = len(weighted) == len(examples)
    regrets = sorted(regret for _, regret in weighted)
    mean = ((math.fsum(weight * regret for weight, regret in weighted) /
             math.fsum(weight for weight, _ in weighted)) if weighted else None)
    return {
        "workloads": len(examples),
        "measured": len(weighted),
        "coverage": len(weighted) / len(examples) if examples else 0,
        "weighted_mean_regret": mean if complete else None,
        "covered_weighted_mean_regret": mean,
        "max_regret": max(regrets) if regrets else None,
        "p95_regret": regrets[math.ceil(0.95 * len(regrets)) - 1] if regrets else None,
        "leaves": tree.leaves,
        "depth": tree.depth,
        "objective": mean + leaf_penalty * tree.leaves if complete and mean is not None else None,
        "per_workload": rows,
    }
