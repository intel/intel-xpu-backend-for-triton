"""Tests for the catalog (synthetic data only)."""
import json
from pathlib import Path

import pytest
import yaml

from benchmark_monitor import CatalogError, SeriesKey, load_catalog
from benchmark_monitor.catalog import DEFAULT_CATALOG_PATH, SCHEMA_PATH


def key(benchmark="gemm", platform="b580", metric="tflops"):
    return SeriesKey("micro", platform, "client", benchmark, "triton", metric, "{}")


def test_default_loads_and_matches_schema_keys():
    cat = load_catalog()
    schema = json.loads(SCHEMA_PATH.read_text())
    assert set(yaml.safe_load(DEFAULT_CATALOG_PATH.read_text())) <= set(schema["properties"])
    assert set(cat.scopes) == {"nightly-b580-micro", "nightly-b580-vllm"}
    assert cat.scopes["nightly-b580-micro"].tags == ("client", )


def test_resolve_inheritance_order():
    cat = load_catalog()
    assert cat.resolve(key()).min_drop_pct == 5.0
    assert cat.resolve(key("softmax")).min_drop_pct == 3.0
    assert cat.resolve(key("batched-flash-attn-bwd")).min_drop_pct == 8.0
    t = cat.resolve(key("flex-attn-causal", metric="time_us"))
    assert (t.min_drop_pct, t.higher_is_better, t.lock_mad_frac, t.window) == (8.0, False, 0.05, 20)


def test_platform_overrides_benchmark():
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace('b580: {match: "B580"}',
                                                                    'b580: {match: "B580", min_drop_pct: 12.0}')
    assert load_catalog(text).resolve(key("softmax")).min_drop_pct == 12.0


def test_normalize_platform():
    cat = load_catalog()
    assert cat.normalize_platform("Intel(R) Arc(TM) B580 Graphics") == "b580"
    assert cat.normalize_platform("Intel Data Center GPU Max 1550") == "max1550"
    assert cat.normalize_platform("something else") is None


@pytest.mark.parametrize("mutation", [
    "bogus: 1\n",
    "defaults: {min_drop_pct: 5.0, zz: 1, z: 3, min_history: 8, window: 20, max_cv_pct: 15, lock_mad_frac: 0.05}\n",
    "benchmarks: {a: {min_drop_pct: -1}}\n",
    "scopes: {s: {source: micro, platform: nope}}\n",
])
def test_invalid_catalogs_raise(mutation):
    base = "version: 1\ndefaults: {min_drop_pct: 5, z: 3, min_history: 8, window: 20, max_cv_pct: 15, lock_mad_frac: 0.05}\n"
    with pytest.raises(CatalogError):
        load_catalog(base + mutation if "defaults" not in mutation else "version: 1\n" + mutation)


def test_strict_validate_unmatched_selector():
    cat = load_catalog()
    cat.validate(["softmax", "gemm", "flash-attn", "flex-attn-causal"], strict=True)
    with pytest.raises(CatalogError, match="matches no known benchmark"):
        cat.validate(["gemm"], strict=True)
    cat.validate(["gemm"])  # non-strict: fine


def test_strict_validate_reaches_family_and_scope_checks():
    with pytest.raises(CatalogError, match="family 'flash-attn'"):
        load_catalog().validate(["softmax"], strict=True)
    text = BASE + METRICS + "platforms: {p: {match: P}}\nscopes: {s: {source: micro, platform: p, benchmarks: ['nope*']}}\n"
    with pytest.raises(CatalogError, match="selector 'nope\\*'"):
        load_catalog(text).validate(["gemm"], strict=True)


def test_overlay_adds_scope_but_not_thresholds(tmp_path: Path):
    ok = "version: 1\nplatforms: {x1: {match: 'X1'}}\nscopes: {s: {source: artifact, platform: x1}}\nimportance: {gemm: 1}\n"
    cat = load_catalog(overlays=[ok])
    assert "x1" in cat.platforms and "s" in cat.scopes and cat.importance == {"gemm": 1}
    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nbenchmarks: {gemm: {min_drop_pct: 1}}\n")
    with pytest.raises(CatalogError, match="may not change"):
        load_catalog(overlays=[p])
    with pytest.raises(CatalogError, match="only set 'match'"):
        load_catalog(overlays=["version: 1\nplatforms: {x1: {match: 'X', z: 1}}\n"])
    with pytest.raises(CatalogError, match="redefines"):
        load_catalog(overlays=["version: 1\nplatforms: {b580: {match: 'B'}}\n"])


def test_transform_results_metrics_are_declared():
    cat = load_catalog()
    assert not cat.resolve(key(metric="time_ms")).higher_is_better
    assert not cat.resolve(key(metric="ns_per_arg")).higher_is_better
    assert [cat.metric_state(m) for m in ("tflops", "cpu_mhz", "nope")] == ["gated", "ungated", "undeclared"]


def test_undeclared_metric_raises_instead_of_defaulting():
    with pytest.raises(CatalogError, match="not declared"):
        load_catalog().resolve(key(metric="latency_furlongs"))


OVERLAPPING = """version: 1
defaults: {min_drop_pct: 5, z: 3, min_history: 8, window: 20, max_cv_pct: 15, lock_mad_frac: 0.05}
metrics: {tflops: {higher_is_better: true}}
families:
  a: {match: ["flash*"], min_drop_pct: 8}
  b: {match: ["flash-attn*"], min_drop_pct: 12}
"""


def test_first_matching_family_wins_for_thresholds_and_family_of():
    with pytest.warns(UserWarning, match="overlap"):
        cat = load_catalog(OVERLAPPING)
    assert cat.family_of("flash-attn-x") == "a" and cat.resolve(key("flash-attn-x")).min_drop_pct == 8
    assert cat.family_of("gemm") == "gemm"  # no family: its own name


def test_default_catalog_families_do_not_overlap(recwarn):
    load_catalog()
    assert not [w for w in recwarn if "overlap" in str(w.message)]


@pytest.mark.parametrize("bad", [
    "window: 0", "window: 2.5", "min_history: 0", "z: 0", "min_drop_pct: -1", "max_cv_pct: 0", "min_drop_pct: .inf",
    "window: true"
])
def test_threshold_values_are_validated(bad):
    name, value = bad.split(": ")
    base = {
        "min_drop_pct": "5", "z": "3", "min_history": "8", "window": "20", "max_cv_pct": "15", "lock_mad_frac": "0.05"
    }
    base[name] = value
    body = ", ".join(f"{k}: {v}" for k, v in base.items())
    with pytest.raises(CatalogError, match=name):
        load_catalog(f"version: 1\ndefaults: {{{body}}}\n")


@pytest.mark.parametrize("scope", [
    "{source: micro, platform: b580, tags: client}",
    "{source: micro, platform: b580, benchmarks: [1]}",
    "{source: banana, platform: b580}",
])
def test_scope_fields_are_validated(scope):
    base = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").split("scopes:")[0]
    with pytest.raises(CatalogError):
        load_catalog(base + f"scopes:\n  s: {scope}\nimportance: {{}}\n")


def test_bool_version_is_rejected():  # True == 1 in Python; schema.json says const 1
    with pytest.raises(CatalogError, match="unsupported version"):
        load_catalog(BASE.replace("version: 1", "version: true"))


def test_single_line_text_is_yaml_not_a_path():
    with pytest.raises(CatalogError, match="unsupported version"):
        load_catalog("version: 2")
    with pytest.raises(CatalogError, match="expected a mapping"):
        load_catalog("no-such-file.yaml")


def test_max_window_looks_at_every_override():
    assert load_catalog().max_window() == 20
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace("softmax: {min_drop_pct: 3.0}",
                                                                    "softmax: {min_drop_pct: 3.0, window: 35}")
    assert load_catalog(text).max_window() == 35


BASE = "version: 1\ndefaults: {min_drop_pct: 5, z: 3, min_history: 8, window: 20, max_cv_pct: 15, lock_mad_frac: 0.05}\n"
METRICS = "metrics: {tflops: {higher_is_better: true}}\n"


@pytest.mark.parametrize("section", [
    "benchmarks: {gemm: null}",
    "benchmarks: {gemm: 3}",
    "families: {f: null}",
    "families: {f: [a]}",
    "platforms: {p: null}",
    "platforms: {p: [x]}",
    "families: {f: {match: [1]}}",
    "families: {f: {match: ['a', null]}}",
    "families: {f: {match: []}}",
    "families: {f: {match: a*}}",
    "platforms: {p: {match: 1}}",
    "scopes: {s: null}",
    "scopes: {s: {source: [micro], platform: p}}",
    "scopes: {s: {source: micro, platform: [p]}}",
    "scopes: {s: {source: micro, platform: p, expected_nightly: 'false'}}",
    "scopes: {s: {source: micro, platform: p, expected_nightly: 1}}",
    "scopes: {s: {source: micro, platform: p, groups: [null]}}",
    "benchmarks: {123: {}}",
    "benchmarks: {gemm: {1: 2, zz: 3}}",
    "families: {7: {match: [a]}}",
    "platforms: {1: {match: x}}",
    "scopes: {true: {source: micro, platform: p}}",
])
def test_malformed_sections_raise_catalog_error(section):
    with pytest.raises(CatalogError):
        load_catalog(BASE + METRICS + "platforms: {p: {match: P}}\n" * (not section.startswith("platforms")) + section +
                     "\n")


def test_overlay_non_mapping_platform_is_catalog_error():
    with pytest.raises(CatalogError):
        load_catalog(overlays=["version: 1\nplatforms: {x1: null}\n"])


def test_benchmark_inherits_family_then_overrides_it():
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace(
        "softmax: {min_drop_pct: 3.0}", "softmax: {min_drop_pct: 3.0}\n  flash-attn-fwd: {min_drop_pct: 2.0}")
    cat = load_catalog(text)
    assert cat.resolve(key("flash-attn-fwd")).min_drop_pct == 2.0  # benchmark beats its family (8.0)
    assert cat.resolve(key("flash-attn-bwd")).min_drop_pct == 8.0  # sibling keeps the family value


def test_benchmark_inherits_unset_keys_from_family():
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace(
        "flex-attn: {match",
        "flex-attn: {window: 30, match").replace("softmax: {min_drop_pct: 3.0}",
                                                 "softmax: {min_drop_pct: 3.0}\n  flex-attn-x: {z: 4}")
    t = load_catalog(text).resolve(key("flex-attn-x"))
    assert (t.window, t.z, t.min_drop_pct) == (30, 4, 8.0)


def test_strict_benchmark_keys_are_exact_names():
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace("softmax: {min_drop_pct: 3.0}",
                                                                    "'soft*': {min_drop_pct: 3.0}")
    with pytest.warns(UserWarning, match="exact names"):
        cat = load_catalog(text)
    with pytest.warns(UserWarning, match="exact names"), pytest.raises(CatalogError, match="benchmark 'soft\\*'"):
        cat.validate(["softmax", "gemm", "flash-attn", "flex-attn-causal"], strict=True)
    with pytest.warns(UserWarning, match="exact names"):
        load_catalog(text.replace("'soft*'", "'gemm[bf16]'"))


@pytest.mark.parametrize("override", [
    "defaults: {min_drop_pct: 5, z: 3, min_history: 8, window: 5, max_cv_pct: 15, lock_mad_frac: 0.05}",
])
def test_min_history_above_window_is_rejected_in_defaults(override):
    with pytest.raises(CatalogError, match="min_history 8 exceeds window 5"):
        load_catalog("version: 1\n" + override + "\n")


@pytest.mark.parametrize("old, new", [
    ("softmax: {min_drop_pct: 3.0}", "softmax: {window: 5}"),
    ("flex-attn: {match", "flex-attn: {window: 5, match"),
    ('b580: {match: "B580"}', 'b580: {match: "B580", window: 5}'),
])
def test_min_history_above_window_is_rejected_in_overrides(old, new):
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace(old, new)
    with pytest.raises(CatalogError, match="exceeds window"):
        load_catalog(text)


def test_family_reached_only_through_overrides_is_not_checked_alone():
    text = BASE + METRICS + "families: {f: {match: [gemm], window: 5}}\nbenchmarks: {gemm: {min_history: 4}}\n"
    thresholds = load_catalog(text).resolve(key("gemm"))
    assert (thresholds.min_history, thresholds.window) == (4, 5)
    with pytest.raises(CatalogError, match="family 'f'"):
        load_catalog(text.replace("[gemm]", "[gemm, softmax]"))
    with pytest.raises(CatalogError, match="family 'f'"):
        load_catalog(text.replace("[gemm]", "['gemm*']"))


def test_window_and_min_history_overridden_together_are_fine():
    text = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8").replace("softmax: {min_drop_pct: 3.0}",
                                                                    "softmax: {window: 5, min_history: 4}")
    assert load_catalog(text).resolve(key("softmax")).window == 5


def test_strict_validate_warns_on_overlap_missed_by_pairwise_check():
    text = BASE + METRICS + 'families:\n  a: {match: ["flash*"]}\n  b: {match: ["*-bwd"]}\n'
    cat = load_catalog(text)  # "flash*" and "*-bwd" do not match each other as names
    cat.validate(["flash-attn-bwd"])
    with pytest.warns(UserWarning, match="matches families"):
        cat.validate(["flash-attn-bwd"], strict=True)


def test_long_single_line_yaml_text_is_not_a_path():
    pad = ", ".join(f"m{i}: {{higher_is_better: true}}" for i in range(40))
    text = ("{version: 1, defaults: {min_drop_pct: 5, z: 3, min_history: 8, window: 20, max_cv_pct: 15, "
            f"lock_mad_frac: 0.05}}, metrics: {{{pad}}}}}")
    assert len(text) > 300 and "\n" not in text
    assert load_catalog(text).is_gated("m3")


def test_is_gated():
    cat = load_catalog()
    assert cat.is_gated("tflops") is True
    assert cat.is_gated("cpu_mhz") is False
    with pytest.raises(CatalogError, match="not declared"):
        cat.is_gated("latency_furlongs")


def test_schema_agrees_with_loader():
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(json.loads(SCHEMA_PATH.read_text()))
    good = DEFAULT_CATALOG_PATH.read_text(encoding="utf-8")
    assert not list(validator.iter_errors(yaml.safe_load(good)))
    bad = [
        good.replace("softmax: {min_drop_pct: 3.0}", "softmax: {min_drop: 3.0}"),  # typo'd key
        good.replace("flex-attn: {match", "flex-attn: {zz: 1, match"),
        good.replace('b580: {match: "B580"}', 'b580: {match: "B580", zz: 1}'),
        good.replace("  max_cv_pct: 15.0\n", ""),  # threshold missing from defaults
        good.replace("defaults:\n", "defaults_gone:\n"),
        good.replace("expected_nightly: true", "expected_nightly: 'true'", 1),
    ]
    for text in bad:
        assert list(validator.iter_errors(yaml.safe_load(text))), text
        with pytest.raises(CatalogError):
            load_catalog(text)
