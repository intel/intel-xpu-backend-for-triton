# SPDX-License-Identifier: Apache-2.0
"""UA feature/validity contract and dependency-free selector export."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import pprint
import tokenize
from pathlib import Path

from learner import Feature, Tree, canonical, number, predict
from policy_runtime import (
    AVAILABLE_FEATURES,
    CATEGORICAL,
    DEFAULT_FEATURES,
    DYNAMIC_NUMERIC_FEATURES,
    OPTIONAL_FEATURES,  # noqa: F401 - adapter API
    REGIME_FIELDS,
    STATIC_FIELDS,  # noqa: F401 - adapter API
    domain_contains,
    power_of_two,
    valid_config,
)


def feature_specs(names=DEFAULT_FEATURES):
    unknown = set(names) - set(AVAILABLE_FEATURES)
    if unknown:
        raise ValueError(f"Features are not in the UA adapter allowlist: {sorted(unknown)}")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate requested features")
    return tuple(Feature(name, "categorical" if name in CATEGORICAL else "numeric") for name in names)


def adapt_records(rows):
    result = []
    for row in rows:
        if row.get("key") is None:
            raise ValueError(f"{row['id']}: workload has no key; status={row.get('status', 'unknown')}")
        features = {name: value for name, value in row["key"].items() if name in AVAILABLE_FEATURES}
        for name, value in row.get("features", {}).items():
            if name not in AVAILABLE_FEATURES:
                continue
            if name in features and features[name] != value:
                raise ValueError(f"{row['id']}: inconsistent feature {name}")
            features[name] = value
        workload = row.get("workload", {})
        if "seq_lens" in workload:
            total = sum(q for q, _ in workload["seq_lens"])
            count = len(workload["seq_lens"])
            derived = {
                "query_work": total * workload["q_heads"],
                "grid_fill": count * workload["k_heads"],
                "avg_query_len": total / max(count, 1),
            }
            for name, value in derived.items():
                if name in features and features[name] != value:
                    raise ValueError(f"{row['id']}: saved {name} differs from raw workload")
                features[name] = value
        if set(DEFAULT_FEATURES) - features.keys():
            raise ValueError(f"{row['id']}: missing UA features {sorted(set(DEFAULT_FEATURES) - features.keys())}")
        result.append(dict(row, features=features))
    return result


def make_domain(records, specs, policy="supported-values"):
    if policy not in ("supported-values", "observed-regimes"):
        raise ValueError(f"Unknown domain policy: {policy}")
    combinations = {canonical([row["features"][name] for name in REGIME_FIELDS]) for row in records}
    numeric = {
        feature.name: [
            min(row["features"][feature.name]
                for row in records),
            max(row["features"][feature.name]
                for row in records),
        ]
        for feature in specs
        if feature.kind == "numeric" and feature.name not in DYNAMIC_NUMERIC_FEATURES
    }
    categorical = {
        feature.name: sorted({row["features"][feature.name]
                              for row in records}, key=canonical)
        for feature in specs
        if feature.kind == "categorical"
    }
    domain = {
        "policy": policy,
        "supported_values":
        {name: sorted({row["features"][name]
                       for row in records}, key=canonical)
         for name in REGIME_FIELDS},
        "numeric_ranges": numeric,
        "positive_numeric_features": [feature.name for feature in specs if feature.name in DYNAMIC_NUMERIC_FEATURES],
        "categorical_values": categorical,
    }
    if policy == "observed-regimes":
        domain.update(regime_fields=list(REGIME_FIELDS), regimes=[json.loads(value) for value in sorted(combinations)])
    return domain


def select_model(model, features, device_name):
    if device_name != model["target"]["device_name"] or not domain_contains(model["domain"], features):
        return None
    index = predict(Tree.from_dict(model["tree"]), features)
    config = model["configs"][index]
    return dict(config) if valid_config(config, features) else None


def _python_header(model, *, inline=False):
    banner = [
        "# SPDX-License-Identifier: Apache-2.0",
        '"""Generated offline UA selector. None requests the existing heuristic fallback.',
        "",
        "Pass host/capture-time feature values. The caller retains the kernel layout",
        "and quantization eligibility checks and validates the software environment.",
        '"""',
    ]
    if inline:
        banner = [
            "# Generated UA constants and heuristics; insert after the UA module imports.",
            "# None requests the caller's existing heuristic fallback.",
            "# Keep host/capture metadata, eligibility checks, and software compatibility checks.",
            f"# Model SHA256: {hashlib.sha256(canonical(model).encode()).hexdigest()}",
            f"# Exporter SHA256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}",
        ]
    return banner + [
        "import math as _ua_tuned_math" if inline else "import math",
        "",
        f"_DEVICE_NAME = {model['target']['device_name']!r}",
        f"_DOMAIN = {pprint.pformat(model['domain'], width=100, compact=True)}",
    ]


def _guard_sources():
    return [
        inspect.getsource(number),
        inspect.getsource(power_of_two),
        inspect.getsource(valid_config),
        inspect.getsource(domain_contains),
    ]


def export_python(model, *, inline=False):
    """Emit ordinary Python branches with a complete configuration dictionary."""
    lines = _python_header(model, inline=inline) + ["_CONFIGS = {"]
    used = set()

    def collect(node):
        if "config" in node:
            used.add(node["config"])
        else:
            collect(node["left"])
            collect(node["right"])

    collect(model["tree"])
    lines += [f"    {index}: {model['configs'][index]!r}," for index in sorted(used)]
    lines += [
        "}",
        "",
        *_guard_sources(),
        "",
        "def select_config(*, device_name, **features):",
        '    """Return a complete launch config, or None for heuristic fallback."""',
        "    if device_name != _DEVICE_NAME or not domain_contains(_DOMAIN, features):",
        "        return None",
    ]

    def branches(node, indent):
        prefix = "    " * indent
        if "config" in node:
            lines.append(f"{prefix}index = {node['config']}")
        else:
            lines.append(f"{prefix}if features[{node['feature']!r}] {node['operator']} {node['value']!r}:")
            branches(node["left"], indent + 1)
            lines.append(f"{prefix}else:")
            branches(node["right"], indent + 1)

    branches(model["tree"], 1)
    lines += ["    config = _CONFIGS[index]", "    return dict(config) if valid_config(config, features) else None", ""]
    return "\n".join(lines)


def export_inline_python(model):
    """Emit a constant config dictionary and branches for embedding in the UA module."""
    names = {
        "math": "_ua_tuned_math",
        "_DEVICE_NAME": "_UA_TUNED_DEVICE_NAME",
        "_DOMAIN": "_UA_TUNED_DOMAIN",
        "_CONFIGS": "_UA_TUNED_CONFIGS",
        "number": "_ua_tuned_number",
        "power_of_two": "_ua_tuned_power_of_two",
        "valid_config": "_ua_tuned_valid_config",
        "domain_contains": "_ua_tuned_domain_contains",
        "select_config": "_select_ua_tuned_config",
    }
    source = export_python(model, inline=True)
    # Token replacement leaves feature names, literals, and dictionary formatting intact.
    tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    for index, token in enumerate(tokens):
        if token.type == tokenize.NAME and token.string in names and tokens[index - 1].string != "import":
            tokens[index] = token._replace(string=names[token.string])
    return tokenize.untokenize(tokens)


def decision_dictionary(model):
    """Map each root-to-leaf conjunction to a complete config, without merging leaves."""
    decisions = {}
    opposite = {"<=": ">", "==": "!="}

    def visit(node, conditions):
        if "config" in node:
            decisions[conditions] = dict(model["configs"][node["config"]])
            return
        feature, operator, value = node["feature"], node["operator"], node["value"]
        if operator not in opposite:
            raise ValueError(f"Unsupported tree operator: {operator}")
        visit(node["left"], conditions + ((feature, operator, value), ))
        visit(node["right"], conditions + ((feature, opposite[operator], value), ))

    visit(model["tree"], ())
    return decisions


def conditions_match(conditions, features):
    """Evaluate a conjunction of exported predicates without parsing Python expressions."""
    for feature, operator, expected in conditions:
        actual = features[feature]
        if operator == "<=":
            matches = actual <= expected
        elif operator == ">":
            matches = actual > expected
        elif operator == "==":
            matches = actual == expected
        elif operator == "!=":
            matches = actual != expected
        else:
            raise ValueError(f"Unsupported decision operator: {operator}")
        if not matches:
            return False
    return True


def export_decisions_json(model):
    """Use named entries because JSON object keys cannot contain condition tuples."""
    return {
        "schema_version": 1,
        "target": model["target"],
        "domain": model["domain"],
        "features": model["features"],
        "fallback": None,
        "provenance": {
            "model_canonical_sha256": hashlib.sha256(canonical(model).encode()).hexdigest(),
            "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "decisions": {
            f"leaf_{index:03d}": {
                "conditions": [{"feature": feature, "operator": operator, "value": value}
                               for feature, operator, value in conditions],
                "config":
                config,
            }
            for index, (conditions, config) in enumerate(decision_dictionary(model).items())
        },
    }


def export_decisions_python(model):
    """Emit a literal condition-tuple dictionary with the same guarded selector API."""
    decisions = decision_dictionary(model)
    lines = _python_header(model) + [
        "# Each key is a conjunction of (feature, operator, value) conditions.",
        f"DECISIONS = {pprint.pformat(decisions, width=100, sort_dicts=False)}",
        "",
        *_guard_sources(),
        inspect.getsource(conditions_match),
        "",
        "def select_config(*, device_name, **features):",
        '    """Select from DECISIONS; return None for the existing heuristic fallback."""',
        "    if device_name != _DEVICE_NAME or not domain_contains(_DOMAIN, features):",
        "        return None",
        "    for conditions, config in DECISIONS.items():",
        "        if conditions_match(conditions, features):",
        "            return dict(config) if valid_config(config, features) else None",
        "    return None",
        "",
    ]
    return "\n".join(lines)
