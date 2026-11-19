# SPDX-License-Identifier: Apache-2.0
"""Independent exhaustive checks for the offline tree optimizer and deployment adapter."""

import ast
import importlib.util
import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from learn import check_export, guarded_report, metadata_context, split_records
from learn import main as learner_main
from learner import Dataset, ExactLearner, Example, Feature, Tree, canonical, evaluate, predict
from ua_adapter import (
    AVAILABLE_FEATURES,
    DEFAULT_FEATURES,
    OPTIONAL_FEATURES,
    adapt_records,
    conditions_match,
    decision_dictionary,
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

CONFIGS = [{"choice": name} for name in ("a", "b", "c")]
FEATURES = (Feature("x"), Feature("y"))


def records(values):
    return [{
        "id":
        str(index),
        "features": {"x": x, "y": y},
        "weight":
        weight,
        "options": [{"config": CONFIGS[c], "status": "ok", "samples_ms": [latency] * 3}
                    for c, latency in enumerate(times)
                    if latency is not None],
    }
            for index, (x, y, weight, times) in enumerate(values)]


def brute_force(data, depth, penalty):
    """Enumerate all binary threshold trees, independently computing leaf regret."""
    total_weight = sum(row.weight for row in data.examples)

    def recurse(rows, remaining):
        best = math.inf
        for identifier in data.config_ids:
            if all(identifier in row.latencies for row in rows):
                value = sum(row.weight / total_weight * (row.latencies[identifier] / min(row.latencies.values()) - 1)
                            for row in rows)
                best = min(best, value + penalty)
        if remaining:
            for feature in data.features:
                values = sorted({row.features[feature.name] for row in rows})
                for value in values[:-1]:
                    left = [row for row in rows if row.features[feature.name] <= value]
                    right = [row for row in rows if row.features[feature.name] > value]
                    best = min(best, recurse(left, remaining - 1) + recurse(right, remaining - 1))
        return best

    return recurse(data.examples, depth)


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
@pytest.mark.parametrize("penalty", [0, 0.015, 0.4])
def test_dp_matches_exhaustive_trees(depth, penalty):
    generator = random.Random(42)
    for _ in range(12):
        rows = records([(
            x,
            y,
            generator.randint(1, 9),
            [
                generator.uniform(0.1, 8),
                generator.uniform(0.1, 8),
                generator.uniform(0.1, 8) if generator.random() > 0.3 else None,
            ],
        ) for x, y in itertools.product(range(2), range(2))])
        data = Dataset.from_records(rows, FEATURES)
        tree, _, stats = ExactLearner(data, depth, penalty).fit()
        expected = brute_force(data, depth, penalty)
        assert stats["optimality_proven"]
        assert stats["objective"] == pytest.approx(expected, abs=1e-12)
        assert evaluate(tree, data, leaf_penalty=penalty)["objective"] == pytest.approx(expected)


def test_exact_search_finds_split_without_immediate_gain():
    rows = records([(x, y, 1, [1, 2] if x == y else [2, 1]) for x, y in itertools.product(range(2), repeat=2)])
    data = Dataset.from_records(rows, FEATURES)
    tree, greedy, stats = ExactLearner(data, 2, 0.01).fit()
    assert greedy.leaves == 1
    assert tree.leaves == 4
    assert stats["objective"] == pytest.approx(0.04)
    assert stats["greedy_objective"] == pytest.approx(0.51)


def test_shared_config_can_win_without_winning_any_workload():
    data = Dataset.from_records(records([(0, 0, 1, [5, 100, 6]), (1, 0, 1, [100, 5, 6])]), FEATURES)
    tree, _, _ = ExactLearner(data, 0, 0.01).fit()
    assert data.configs[tree.config] == CONFIGS[2]
    assert evaluate(tree, data)["weighted_mean_regret"] == pytest.approx(0.2)


def test_missing_pairs_require_a_split():
    data = Dataset.from_records(records([(0, 0, 1, [1, None]), (1, 0, 1, [None, 1])]), FEATURES)
    with pytest.raises(ValueError, match="No feasible tree"):
        ExactLearner(data, 0).fit()
    tree, _, stats = ExactLearner(data, 1).fit()
    assert tree.leaves == 2
    assert stats["objective"] == pytest.approx(0.004)
    assert evaluate(tree, data)["coverage"] == 1


def test_indistinguishable_rows_keep_their_individual_costs():
    data = Dataset.from_records(records([(0, 0, 1, [1, 2]), (0, 0, 9, [100, 1])]), FEATURES)
    assert len(data.groups) == 1
    tree, _, _ = ExactLearner(data, 3).fit()
    assert data.configs[tree.config] == CONFIGS[1]
    assert evaluate(tree, data)["weighted_mean_regret"] == pytest.approx(0.1)
    bad = records([(0, 0, 1, [1, None]), (0, 0, 1, [None, 1])])
    with pytest.raises(ValueError, match="Indistinguishable"):
        Dataset.from_records(bad, FEATURES)


def test_categorical_and_numeric_predicates():
    rows = records([(0, 0, 1, [1, 5]), (1, 0, 1, [5, 1])])
    rows[0]["features"]["x"] = "prefill"
    rows[1]["features"]["x"] = "decode"
    data = Dataset.from_records(rows, [Feature("x", "categorical")])
    tree, _, _ = ExactLearner(data, 1, 0).fit()
    assert tree.predicate.operator == "=="
    assert evaluate(tree, data)["weighted_mean_regret"] == 0
    data = Dataset.from_records(records([(64, 0, 1, [1, 5]), (128, 0, 1, [5, 1])]), FEATURES)
    tree, _, _ = ExactLearner(data, 1, 0).fit()
    assert tree.predicate.value == 96
    assert predict(tree, {"x": 96}) != predict(tree, {"x": math.nextafter(96, math.inf)})
    assert Tree.from_dict(tree.to_dict()) == tree


@pytest.mark.parametrize("limits,reason", [({"time_limit": 0}, "time_limit"), ({"max_states": 0}, "state_limit")])
def test_limits_retain_complete_greedy_incumbent(limits, reason):
    data = Dataset.from_records(records([(0, 0, 1, [1, 2]), (1, 0, 1, [2, 1])]), FEATURES)
    tree, greedy, stats = ExactLearner(data, 3, **limits).fit()
    assert tree == greedy
    assert not stats["optimality_proven"]
    assert stats["stopping_reason"] == reason
    assert evaluate(tree, data)["coverage"] == 1


def test_failed_or_incorrect_options_are_never_selected():
    rows = records([(0, 0, 1, [10, 1, 0.1])])
    rows[0]["options"][1]["status"] = "compilation_error"
    rows[0]["options"][2]["correctness"] = "failed"
    data = Dataset.from_records(rows, FEATURES)
    tree, _, _ = ExactLearner(data, 0).fit()
    assert data.configs[tree.config] == CONFIGS[0]
    with pytest.raises(ValueError, match="no usable"):
        Dataset.from_records(rows, FEATURES, require_correctness=True)
    rows[0]["options"][0]["correctness"] = "passed"
    assert Dataset.from_records(rows, FEATURES, require_correctness=True)


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_reject_invalid_measurements(latency):
    rows = records([(0, 0, 1, [1])])
    rows[0]["options"][0]["samples_ms"] = [latency]
    with pytest.raises(ValueError, match="timing"):
        Dataset.from_records(rows, FEATURES)


def test_unmeasured_validation_config_does_not_receive_a_synthetic_timing():
    training = Dataset.from_records(records([(0, 0, 1, [1, 10])]), FEATURES)
    test = Dataset.from_records(records([(1, 0, 1, [None, 1])]), FEATURES)
    tree, _, _ = ExactLearner(training, 0).fit()
    result = evaluate(tree, training, test.examples)
    assert result["weighted_mean_regret"] is None and result["coverage"] == 0
    assert result["per_workload"][0]["regret"] is None


def ua_rows():
    config = {"block_m": 32, "tile_size": 16, "num_warps": 4, "num_stages": 1, "grf_mode": "256"}
    key = {
        "block_size": 16,
        "head_size": 128,
        "num_queries_per_kv": 7,
        "use_td": True,
        "use_fp8_kv": False,
        "sliding_window": 0,
        "q_element_size": 2,
        "k_element_size": 2,
        "v_element_size": 2,
        "out_element_size": 2,
        "is_3d": False,
        "q_len_bucket": 3,
        "num_seqs": 1,
    }
    return adapt_records([{
        "id": str(index),
        "key": dict(key, query_work=q * 28),
        "workload": {"q_heads": 28, "k_heads": 4, "seq_lens": [[q, q]]},
        "options": [{"config": config, "status": "ok", "samples_ms": [1, 1.1, 1.2]}],
    } for index, q in enumerate((4096, 8192))])


def test_ua_export_guards_and_non_power_of_two_gqa(tmp_path):
    rows = ua_rows()
    specs = feature_specs()
    data = Dataset.from_records(rows, specs, valid_config)
    tree, _, _ = ExactLearner(data, 1).fit()
    model = {
        "tree": tree.to_dict(),
        "configs": list(data.configs),
        "domain": make_domain(rows, specs),
        "target": {"device_name": "B580"},
        "features": [{"name": f.name, "kind": f.kind} for f in specs],
    }
    path = tmp_path / "selector.py"
    path.write_text(export_python(model))
    spec = importlib.util.spec_from_file_location("generated_ua_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    select = module.select_config
    assert check_export(model, rows) > len(rows)
    assert select(device_name="B580", **rows[0]["features"]) == data.configs[0]
    assert select(device_name="other", **rows[0]["features"]) is None
    assert select(device_name="B580", **dict(rows[0]["features"], head_size=256)) is None
    assert select(device_name="B580", **dict(rows[0]["features"], query_work=math.nan)) is None
    assert select(device_name="B580", **dict(rows[0]["features"], query_work=1)) == data.configs[0]
    assert select(device_name="B580", **dict(rows[0]["features"], query_work=0)) is None
    assert not valid_config(dict(data.configs[0], tile_size=64), rows[0]["features"])
    assert not valid_config(dict(data.configs[0], block_m=4), rows[0]["features"])
    assert select_model(model, rows[0]["features"], "B580") == select(device_name="B580", **rows[0]["features"])
    # Callers receive independent dictionaries.
    result = select(device_name="B580", **rows[0]["features"])
    result["block_m"] = 0
    assert select(device_name="B580", **rows[0]["features"])["block_m"] == 32


def test_holdout_keeps_identical_feature_vectors_together():
    rows = ua_rows()
    rows += [dict(row, id=row["id"] + "duplicate") for row in rows]
    train, test = split_records(rows, feature_specs(), 0.5, 42)
    assert len(train) == len(test) == 2
    assert {canonical(row["features"]) for row in train}.isdisjoint(canonical(row["features"]) for row in test)


def test_mixed_environment_metadata_is_rejected():
    info = {
        "hardware": {"name": "B580"},
        "software": {},
        "sources": {name: {"sha256": "abc"}
                    for name in ("kernel", "runtime_config", "triton_compiler")},
        "timing": {},
    }
    other = dict(info, hardware={"name": "other GPU"})
    with pytest.raises(ValueError, match="mix different"):
        metadata_context({"per_workload": [{"metadata": info}, {"metadata": other}]})


def test_saved_samples_are_reduced_by_median_not_minimum():
    rows = records([(0, 0, 1, [1])])
    rows[0]["options"][0]["samples_ms"] = [1, 4, 100]
    data = Dataset.from_records(rows, FEATURES)
    assert next(iter(data.examples[0].latencies.values())) == 4


def test_branch_and_bound_prunes_using_admissible_leaf_penalties():
    data = Dataset.from_records(records([(0, 0, 1, [1, 4]), (1, 1, 1, [1, 4])]), FEATURES)
    tree, _, stats = ExactLearner(data, 3, 0.4).fit()
    assert tree.leaves == 1
    assert stats["pruned"] > 0
    assert stats["objective"] == pytest.approx(0.4)


def test_atomic_bound_with_conflicting_duplicate_winners():
    rows = records([(0, 0, 1, [1, 3]), (0, 0, 2, [2, 1]), (1, 0, 3, [5, 1]), (1, 1, 1, [1, 5])])
    data = Dataset.from_records(rows, FEATURES)
    tree, _, stats = ExactLearner(data, 2, 0.03).fit()
    assert stats["objective"] == pytest.approx(brute_force(data, 2, 0.03))
    assert evaluate(tree, data, leaf_penalty=0.03)["objective"] == pytest.approx(stats["objective"])


def test_equivalent_integer_and_float_features_are_indistinguishable():
    rows = records([(1, 0, 1, [1, None]), (1.0, 0, 1, [None, 1])])
    with pytest.raises(ValueError, match="Indistinguishable"):
        Dataset.from_records(rows, FEATURES)


@pytest.mark.parametrize(
    "argument",
    [
        {"leaf_penalty": -1},
        {"leaf_penalty": float("nan")},
        {"max_depth": -1},
        {"time_limit": float("inf")},
        {"max_states": -1},
    ],
)
def test_invalid_search_settings(argument):
    data = Dataset.from_records(records([(0, 0, 1, [1])]), FEATURES)
    with pytest.raises(ValueError):
        ExactLearner(data, **argument)


def unseen_regime_fixture(policy="supported-values"):
    rows = ua_rows()
    rows[0]["features"].update(block_size=16, num_queries_per_kv=4, q_len_bucket=2)
    rows[1]["features"].update(block_size=64, num_queries_per_kv=8, q_len_bucket=3, head_size=256, sliding_window=8192)
    specs = feature_specs()
    data = Dataset.from_records(rows, specs, valid_config)
    tree, _, _ = ExactLearner(data, 1).fit()
    model = {
        "tree": tree.to_dict(),
        "configs": list(data.configs),
        "domain": make_domain(rows, specs, policy),
        "target": {"device_name": "B580"},
        "features": [{"name": f.name, "kind": f.kind} for f in specs],
    }
    unseen = dict(rows[0]["features"], block_size=64)
    return rows, data, model, unseen


@pytest.mark.parametrize("policy,accepted", [("supported-values", True), ("observed-regimes", False)])
def test_new_combinations_follow_explicit_domain_policy(policy, accepted):
    rows, data, model, unseen = unseen_regime_fixture(policy)
    assert model["domain"]["policy"] == policy
    assert domain_contains(model["domain"], unseen) is accepted
    assert select_model(model, unseen, "B580") == (data.configs[0] if accepted else None)
    assert check_export(model, rows + [dict(rows[0], id="held-out", features=unseen)]) > len(rows)


def test_legacy_artifacts_keep_their_exact_regime_guard():
    rows, _, model, unseen = unseen_regime_fixture("observed-regimes")
    del model["domain"]["policy"]
    del model["domain"]["supported_values"]
    assert not domain_contains(model["domain"], unseen)
    assert select_model(model, unseen, "B580") is None
    assert check_export(model, rows + [dict(rows[0], id="held-out", features=unseen)]) > len(rows)


@pytest.mark.parametrize("field,value", [("head_size", 192), ("block_size", 32), ("num_queries_per_kv", 6),
                                         ("sliding_window", 4096)])
def test_supported_values_guard_unseen_static_values_inside_numeric_ranges(field, value):
    rows, _, model, unseen = unseen_regime_fixture()
    low, high = model["domain"]["numeric_ranges"][field]
    assert low < value < high
    outside = dict(unseen, **{field: value})
    assert not domain_contains(model["domain"], outside)
    assert select_model(model, outside, "B580") is None
    assert check_export(model, rows + [dict(rows[0], id="unsupported", features=outside)]) > len(rows)


def test_legality_check_rejects_invalid_config_for_a_supported_combination():
    rows, _, model, unseen = unseen_regime_fixture()
    page16 = dict(unseen, block_size=16)
    assert domain_contains(model["domain"], page16)
    model["tree"] = {"config": 0}
    model["configs"] = [dict(model["configs"][0], tile_size=64)]
    assert not valid_config(model["configs"][0], page16)
    assert select_model(model, page16, "B580") is None
    assert check_export(model, rows) > len(rows)


def test_relaxed_holdout_reports_a_missing_measurement_without_a_fallback():
    _, data, model, unseen = unseen_regime_fixture()
    unseen = dict(unseen, num_seqs=500, query_work=10**12)
    assert domain_contains(model["domain"], unseen)
    # This held-out row measured another valid config, but not the tree's choice.
    other = dict(data.configs[0], num_stages=2)
    example = Example("held-out", unseen, {canonical(other): 1.0})
    tree = Tree.from_dict(model["tree"])
    result = guarded_report(tree, data, [example], model, 0.002)
    assert result["fallbacks"] == 0
    assert result["coverage"] == 0
    assert result["weighted_mean_regret"] is None
    assert result["per_workload"][0]["status"] == "unmeasured_or_invalid"


def test_unknown_domain_policies_are_rejected():
    rows = ua_rows()
    with pytest.raises(ValueError, match="Unknown domain policy"):
        make_domain(rows, feature_specs(), policy="unknown")
    domain = make_domain(rows, feature_specs())
    domain["policy"] = "unknown"
    assert not domain_contains(domain, rows[0]["features"])


def dictionary_fixture():
    rows, _, model, unseen = unseen_regime_fixture()
    model["configs"] = [model["configs"][0], dict(model["configs"][0], num_stages=2)]
    model["tree"] = {
        "feature": "query_work",
        "operator": "<=",
        "value": 172032.0,
        "left": {"config": 0},
        "right": {
            "feature": "q_len_bucket",
            "operator": "==",
            "value": 2,
            "left": {"config": 1},
            "right": {"config": 0},
        },
    }
    return rows, model, unseen


def test_dictionary_export_preserves_both_sides_of_numeric_and_categorical_splits(tmp_path):
    rows, model, unseen = dictionary_fixture()
    decisions = decision_dictionary(model)
    assert decisions == {
        (("query_work", "<=", 172032.0), ): model["configs"][0],
        (("query_work", ">", 172032.0), ("q_len_bucket", "==", 2)): model["configs"][1],
        (("query_work", ">", 172032.0), ("q_len_bucket", "!=", 2)): model["configs"][0],
    }
    path = tmp_path / "decisions.py"
    path.write_text(export_decisions_python(model))
    spec = importlib.util.spec_from_file_location("generated_ua_dictionary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DECISIONS == decisions
    extra = []
    for work, bucket, config in (
        (114688, 2, 0),
        (172032, 3, 0),
        (math.nextafter(172032, math.inf), 2, 1),
        (229376, 2, 1),
        (229376, 3, 0),
    ):
        features = dict(unseen, query_work=work, q_len_bucket=bucket)
        matches = [choice for conditions, choice in decisions.items() if conditions_match(conditions, features)]
        assert len(matches) == 1 and matches[0] == model["configs"][config]
        assert module.select_config(device_name="B580", **features) == model["configs"][config]
        extra.append(dict(rows[0], id=f"boundary-{len(extra)}", features=features))
    assert check_export(model, rows + extra) > len(rows + extra)
    first = module.select_config(device_name="B580", **unseen)
    first["block_m"] = 0
    assert module.select_config(device_name="B580", **unseen)["block_m"] == 32


def test_json_dictionary_round_trip_preserves_conditions_and_complete_configs():
    _, model, _ = dictionary_fixture()
    payload = json.loads(json.dumps(export_decisions_json(model), allow_nan=False))
    reconstructed = {
        tuple((condition["feature"], condition["operator"], condition["value"]) for condition in decision["conditions"]):
        decision["config"]
        for decision in payload["decisions"].values()
    }
    assert reconstructed == decision_dictionary(model)
    assert payload["fallback"] is None
    assert payload["domain"] == model["domain"]
    assert len(payload["provenance"]["model_canonical_sha256"]) == 64
    # Two leaves can choose the same config; each decision region is retained.
    assert len(payload["decisions"]) == 3


def test_dictionary_export_handles_a_tree_with_one_leaf():
    rows, _, model, _ = unseen_regime_fixture()
    assert model["tree"].get("config") is not None
    assert decision_dictionary(model) == {(): model["configs"][0]}
    payload = export_decisions_json(model)
    assert payload["decisions"]["leaf_000"]["conditions"] == []
    assert check_export(model, rows) > len(rows)


def test_inline_constants_embed_without_overwriting_host_module_names():
    rows, model, unseen = dictionary_fixture()
    source = export_inline_python(model)
    syntax = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in syntax.body
        if isinstance(node, ast.Assign)
    }
    assert assignments["_UA_TUNED_CONFIGS"] == dict(enumerate(model["configs"]))
    assert assignments["_UA_TUNED_DOMAIN"] == model["domain"]
    assert assignments["_UA_TUNED_DEVICE_NAME"] == "B580"
    assert [node.names[0].name for node in syntax.body if isinstance(node, ast.Import)] == ["math"]
    assert not any(isinstance(node, ast.ImportFrom) for node in syntax.body)
    assert not any(isinstance(node, ast.Expr) for node in syntax.body)

    host = {
        name: object()
        for name in (
            "math",
            "number",
            "power_of_two",
            "valid_config",
            "domain_contains",
            "select_config",
            "_DEVICE_NAME",
            "_DOMAIN",
            "_CONFIGS",
            "__doc__",
        )
    }
    namespace = dict(host)
    exec(compile(source, "<inline constants inside UA module>", "exec"), namespace)  # noqa: S102
    assert all(namespace[name] is value for name, value in host.items())
    new_names = namespace.keys() - host.keys() - {"__builtins__"}
    assert all(name.startswith(("_UA_TUNED_", "_ua_tuned_")) or name == "_select_ua_tuned_config" for name in new_names)
    selector = namespace["_select_ua_tuned_config"]
    for features in (rows[0]["features"], rows[1]["features"], unseen):
        assert selector(device_name="B580", **features) == select_model(model, features, "B580")
    assert selector(device_name="other", **unseen) is None
    assert selector(device_name="B580", **dict(unseen, query_work=math.nan)) is None
    result = selector(device_name="B580", **unseen)
    result["block_m"] = 0
    assert selector(device_name="B580", **unseen) == model["configs"][0]


@pytest.mark.parametrize("removed", ["num_segments", "kv_len_bucket"])
def test_removed_key_fields_are_ignored_in_legacy_measurements(removed):
    rows = ua_rows()
    legacy = [
        dict(row, key=dict(row["key"], **{removed: 16}), features=dict(row["features"], **{removed: 32}))
        for row in rows
    ]
    adapted = adapt_records(legacy)
    assert [row["features"] for row in adapted] == [row["features"] for row in rows]
    assert removed not in DEFAULT_FEATURES
    assert all(removed not in row["features"] for row in adapted)
    domain = make_domain(adapted, feature_specs())
    assert all(removed not in domain[name] for name in ("supported_values", "numeric_ranges", "categorical_values"))
    with pytest.raises(ValueError, match="not in the UA adapter allowlist"):
        feature_specs([removed])
    # Adaptation preserves the original benchmark evidence.
    assert legacy[0]["key"][removed] == 16
    assert legacy[0]["features"][removed] == 32


@pytest.mark.parametrize("optional", OPTIONAL_FEATURES)
def test_default_exports_ignore_optional_features(optional):
    rows, data, model, unseen = unseen_regime_fixture()
    assert {"query_work", "num_queries_per_kv"}.issubset(DEFAULT_FEATURES)
    assert optional not in DEFAULT_FEATURES and optional in AVAILABLE_FEATURES
    assert optional not in {feature.name for feature in feature_specs()}
    assert all(optional not in model["domain"][name]
               for name in ("numeric_ranges", "categorical_values", "supported_values"))
    features = {name: value for name, value in unseen.items() if name not in OPTIONAL_FEATURES}
    expected = data.configs[0]
    assert select_model(model, features, "B580") == expected
    for exporter, function in (
        (export_python, "select_config"),
        (export_decisions_python, "select_config"),
        (export_inline_python, "_select_ua_tuned_config"),
    ):
        code = exporter(model)
        assert all(name not in code for name in OPTIONAL_FEATURES)
        namespace = {}
        exec(compile(code, "<default selector without optional inputs>", "exec"), namespace)  # noqa: S102
        selector = namespace[function]
        assert selector(device_name="B580", **features) == expected
        for value in (10**30, math.nan):
            assert selector(device_name="B580", **dict(features, **{optional: value})) == expected
    opted_in = feature_specs(DEFAULT_FEATURES + OPTIONAL_FEATURES)
    domain = make_domain(rows, opted_in)
    assert set(OPTIONAL_FEATURES).issubset(domain["positive_numeric_features"])
    assert set(OPTIONAL_FEATURES).isdisjoint(domain["numeric_ranges"])
    assert not domain_contains(domain, features)
    assert {feature.name for feature in opted_in} == set(AVAILABLE_FEATURES)


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_defaults_and_optional_feature_opt_in(explicit, monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("learn.train", lambda args: captured.update(vars(args)))
    argv = [str(tmp_path / "measurements.json"), "--output-dir", str(tmp_path / "model")]
    if explicit:
        argv += ["--features", *AVAILABLE_FEATURES]
    learner_main(argv)
    assert captured["features"] == list(AVAILABLE_FEATURES if explicit else DEFAULT_FEATURES)


def test_default_training_accepts_records_without_optional_metadata():
    rows = [dict(row, workload={}, features={}) for row in ua_rows()]
    adapted = adapt_records(rows)
    assert all(set(OPTIONAL_FEATURES).isdisjoint(row["features"]) for row in adapted)
    data = Dataset.from_records(adapted, feature_specs(), valid_config)
    assert len(data.examples) == len(rows)
    with pytest.raises(ValueError, match="invalid/missing feature grid_fill"):
        Dataset.from_records(adapted, feature_specs(AVAILABLE_FEATURES), valid_config)


def exported_selectors(model):
    result = []
    for exporter, name in (
        (export_python, "select_config"),
        (export_decisions_python, "select_config"),
        (export_inline_python, "_select_ua_tuned_config"),
    ):
        namespace = {}
        exec(compile(exporter(model), "<extrapolating UA selector>", "exec"), namespace)  # noqa: S102
        result.append(namespace[name])
    return result


@pytest.mark.parametrize("policy", ["supported-values", "observed-regimes"])
@pytest.mark.parametrize("field", ["num_seqs", "query_work", "grid_fill", "avg_query_len"])
def test_dynamic_numeric_features_follow_end_branches_beyond_training(field, policy):
    rows, _, model, _ = unseen_regime_fixture(policy)
    if field == "num_seqs":
        rows[0]["features"][field], rows[1]["features"][field] = 16, 256
    elif field == "grid_fill":
        rows[0]["features"][field], rows[1]["features"][field] = 4, 2048
    low, high = (row["features"][field] for row in rows)
    specs = feature_specs(AVAILABLE_FEATURES if field in OPTIONAL_FEATURES else DEFAULT_FEATURES)
    model["features"] = [{"name": spec.name, "kind": spec.kind} for spec in specs]
    model["domain"] = make_domain(rows, specs, policy)
    model["configs"] = [model["configs"][0], dict(model["configs"][0], num_stages=2)]
    model["tree"] = {
        "feature": field,
        "operator": "<=",
        "value": (low + high) / 2,
        "left": {"config": 0},
        "right": {"config": 1},
    }
    assert field not in model["domain"]["numeric_ranges"]
    assert field in model["domain"]["positive_numeric_features"]
    selectors = exported_selectors(model)
    above_training = 500 if field == "num_seqs" else high * 2
    assert above_training > high
    for value, chosen in ((1, 0), (low, 0), (high, 1), (above_training, 1), (10**12, 1)):
        features = dict(rows[0]["features"], **{field: value})
        original = dict(features)
        assert domain_contains(model["domain"], features)
        expected = model["configs"][chosen]
        assert select_model(model, features, "B580") == expected
        for selector in selectors:
            assert selector(device_name="B580", **features) == expected
        assert features == original
    assert check_export(model, rows) > len(rows)


@pytest.mark.parametrize("field", ["num_seqs", "query_work", "grid_fill", "avg_query_len"])
def test_dynamic_numeric_features_reject_invalid_inputs(field):
    rows, _, model, _ = unseen_regime_fixture()
    specs = feature_specs(AVAILABLE_FEATURES)
    model["features"] = [{"name": spec.name, "kind": spec.kind} for spec in specs]
    model["domain"] = make_domain(rows, specs)
    selectors = exported_selectors(model)
    for value in (0, -1, math.nan, math.inf, -math.inf, True, "500", None, []):
        features = dict(rows[0]["features"], **{field: value})
        assert not domain_contains(model["domain"], features)
        assert select_model(model, features, "B580") is None
        for selector in selectors:
            assert selector(device_name="B580", **features) is None
    missing = {name: value for name, value in rows[0]["features"].items() if name != field}
    assert not domain_contains(model["domain"], missing)
    for selector in selectors:
        assert selector(device_name="B580", **missing) is None


def test_legacy_dynamic_numeric_ranges_remain_enforced():
    rows, _, model, _ = unseen_regime_fixture()
    del model["domain"]["positive_numeric_features"]
    for field in ("num_seqs", "query_work"):
        values = [row["features"][field] for row in rows]
        model["domain"]["numeric_ranges"][field] = [min(values), max(values)]
    selectors = exported_selectors(model)
    for changes in ({"num_seqs": 500}, {"query_work": 1}, {"query_work": 10**12}):
        features = dict(rows[0]["features"], **changes)
        assert not domain_contains(model["domain"], features)
        assert select_model(model, features, "B580") is None
        for selector in selectors:
            assert selector(device_name="B580", **features) is None
    assert check_export(model, rows) > len(rows)


def policy_fixture(rows):
    """A fixed tree deliberately choosing the first config, for independent anchor checks."""
    return {
        "schema_version": 1,
        "adapter": "ua_full_gqa_v1",
        "tree": {"config": 0},
        "configs": [option["config"] for option in rows[0]["options"]],
        "domain": make_domain(rows, feature_specs()),
        "target": {"device_name": "B580", "sources": {"kernel": "test-kernel"}},
        "features": [{"name": f.name, "kind": f.kind} for f in feature_specs()],
        "settings": {"require_correctness": False},
        "objective": {"name": "weighted_mean_normalized_regret_plus_leaf_penalty"},
        "provenance": {"inputs": []},
        "training_ids": [row["id"] for row in rows],
        "validation_ids": [],
    }


def test_json_policy_shares_configs_and_handles_colliding_keys(tmp_path):
    from policy import CompiledPolicy, build_policy

    rows = ua_rows()
    first = rows[0]["options"][0]["config"]
    second = dict(first, num_warps=8)
    rows[1]["features"] = dict(rows[0]["features"])
    for row, times, weight in zip(rows, ((1, 2), (4, 1)), (1, 3)):
        row["weight"] = weight
        row["options"] = [{"config": config, "status": "ok", "samples_ms": [latency] * 3}
                          for config, latency in zip((first, second), times)]
    model = policy_fixture(rows)
    model["configs"] = list(reversed(model["configs"]))
    model["tree"] = {"config": 1}  # Remapping must still select the first config on a miss.
    artifact = build_policy(model, rows)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(artifact))
    policy = CompiledPolicy.load(path, expected_target=model["target"])
    assert len(artifact["anchors"]) == 1 and len(artifact["configs"]) == 2
    exact = policy.select(rows[0]["features"], device_name="B580")
    assert exact.source == "exact" and exact.config == second
    unseen = dict(rows[0]["features"], num_seqs=500, query_work=10**12)
    prediction = policy.select(unseen, device_name="B580")
    assert prediction.source == "tree" and prediction.config == first
    assert policy.select(unseen, device_name="other").config is None
    assert policy.select(dict(unseen, head_size=256), device_name="B580").config is None
    exact.config["num_warps"] = 999
    assert policy.select(rows[0]["features"], device_name="B580").config == second


def test_json_policy_joint_distance_is_not_sequential_and_radius_falls_back():
    from policy import CompiledPolicy, build_policy

    rows = ua_rows()
    for row, n, work in zip(rows, (8, 16), (1024, 64)):
        row["features"].update(num_seqs=n, query_work=work)
    model = policy_fixture(rows)
    artifact = build_policy(model, rows)
    artifact["on_miss"] = {"method": "joint-nearest"}
    policy = CompiledPolicy(artifact, expected_target=model["target"])
    query = dict(rows[0]["features"], num_seqs=8, query_work=64)
    selected = policy.select(query, device_name="B580")
    assert selected.source == "joint-nearest" and selected.distance == pytest.approx(1)
    # Sequential selection would choose n=8 first and therefore work=1024.
    assert dict(zip(policy.key_fields, selected.anchor))["num_seqs"] == 16
    artifact["on_miss"]["max_distance"] = 0.5
    restricted = CompiledPolicy(artifact, expected_target=model["target"])
    assert restricted.select(query, device_name="B580").source == "tree"
    assert restricted.select(rows[0]["features"], device_name="B580").source == "exact"
    assert policy.select(dict(query, num_seqs=0), device_name="B580").source == "fallback"


def test_json_policy_retains_tree_generalization_across_unseen_regimes():
    from policy import CompiledPolicy, build_policy

    rows = ua_rows()
    rows[1]["features"].update(num_queries_per_kv=8, block_size=64)
    model = policy_fixture(rows)
    artifact = build_policy(model, rows)
    artifact["on_miss"] = {"method": "joint-nearest"}
    policy = CompiledPolicy(artifact, expected_target=model["target"])
    query = dict(rows[0]["features"], num_queries_per_kv=8)
    assert policy.select(query, device_name="B580").source == "tree"
    # All selected configs are checked even when a JSON tree is edited after export.
    artifact["configs"][artifact["tree"]["config"]]["block_m"] = 999
    artifact["anchors"] = []
    edited = CompiledPolicy(artifact, expected_target=model["target"])
    assert edited.select(query, device_name="B580").config is None


def test_json_policy_rejects_data_leakage_and_malformed_contracts():
    from copy import deepcopy

    from policy import CompiledPolicy, build_policy

    rows = ua_rows()
    model = policy_fixture(rows)
    with pytest.raises(ValueError, match="training IDs"):
        build_policy(model, rows[:1])
    model["validation_ids"] = [rows[0]["id"]]
    with pytest.raises(ValueError, match="Validation"):
        build_policy(model, rows)
    model["validation_ids"] = []
    artifact = build_policy(model, rows)
    with pytest.raises(ValueError, match="target"):
        CompiledPolicy(artifact, expected_target={"device_name": "other"})
    mutations = [
        lambda x: x.update(schema_version=999),
        lambda x: x["anchors"].append(x["anchors"][0]),
        lambda x: x["tree"].update(config=999),
        lambda x: x.update(on_miss={"method": "joint-nearest", "weights": {"num_seqs": 1}}),
        lambda x: x.update(on_miss={"method": "joint-nearest", "max_distance": -1}),
    ]
    for mutate in mutations:
        malformed = deepcopy(artifact)
        mutate(malformed)
        with pytest.raises(ValueError):
            CompiledPolicy(malformed, expected_target=model["target"])


def test_learner_cli_emits_shared_policy_without_validation_anchors(monkeypatch, tmp_path):
    from policy import CompiledPolicy

    rows = ua_rows()
    context = policy_fixture(rows)["target"]
    monkeypatch.setattr("learn.load_inputs", lambda paths: (rows, context, []))
    output = tmp_path / "learned"
    learner_main(["unused.json", "--output-dir", str(output), "--validation-fraction", "0.5"])
    policy = CompiledPolicy.load(output / "policy.json", expected_target=context)
    model = json.loads((output / "tree.json").read_text())
    training_ids = set(model["training_ids"])
    for row in rows:
        expected_source = "exact" if row["id"] in training_ids else "tree"
        assert policy.select(row["features"], device_name="B580").source == expected_source
    assert len(policy.configs) == 1
    assert set(policy.exact.values()) == {policy.tree["config"]}


def test_combined_policy_report_preserves_unmeasured_and_fallback_cases():
    from policy import build_policy, evaluate_policy

    rows = ua_rows()
    artifact = build_policy(policy_fixture(rows), rows)
    measured = Dataset.from_records(rows, feature_specs(), valid_config)
    result = evaluate_policy(artifact, measured.examples)
    assert result["source_counts"] == {"exact": 2}
    assert result["coverage"] == 1 and result["weighted_mean_regret"] == 0
    other = dict(artifact["configs"][0], num_stages=3)
    unseen = dict(rows[0]["features"], num_seqs=500)
    unmeasured = Example("unmeasured", unseen, {canonical(other): 1.0})
    unsupported = Example("unsupported", dict(unseen, head_size=256), {canonical(other): 1.0})
    result = evaluate_policy(artifact, [unmeasured, unsupported])
    assert result["coverage"] == 0 and result["weighted_mean_regret"] is None
    assert result["source_counts"] == {"tree": 1, "fallback": 1}
    assert [row["status"] for row in result["per_workload"]] == ["unmeasured", "fallback"]
