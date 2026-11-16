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
