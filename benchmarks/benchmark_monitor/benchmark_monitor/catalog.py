"""Benchmark catalog: thresholds, platform normalization and monitoring scopes.

Validated at load time (pyyaml only); schema.json is shipped for editors.
"""

from __future__ import annotations

import fnmatch
import math
import re
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from benchmark_monitor.model import SeriesKey, Thresholds

DEFAULT_CATALOG_PATH = Path(__file__).parent / "catalog" / "default.yaml"
SCHEMA_PATH = Path(__file__).parent / "catalog" / "schema.json"

_REQUIRED_KEYS = {"min_drop_pct", "z", "min_history", "window", "max_cv_pct", "lock_mad_frac"}
# The lock-in knobs are optional in `defaults`; Thresholds supplies the #6491 values (8.0 / 8).
_THRESHOLD_KEYS = _REQUIRED_KEYS | {"improvement_lock_pct", "lock_window"}
_TOP_KEYS = {"version", "defaults", "metrics", "platforms", "families", "benchmarks", "scopes", "importance"}
_SCOPE_KEYS = {"source", "platform", "tags", "benchmarks", "groups", "expected_nightly"}
_SOURCE_KINDS = {"micro", "third_party", "artifact"}
_METRIC_KEYS = {"higher_is_better", "gate"}
# key -> (integer?, minimum, minimum is exclusive?)
_RULES: dict[str, tuple[bool, float, bool]] = {
    "min_drop_pct": (False, 0, False),
    "z": (False, 0, True),
    "min_history": (True, 1, False),
    "window": (True, 1, False),
    "max_cv_pct": (False, 0, True),
    "lock_mad_frac": (False, 0, False),
    "improvement_lock_pct": (False, 0, False),
    "lock_window": (True, 1, False),
}
# An overlay may only add these sections; everything else could change thresholds.
_OVERLAY_KEYS = {"version", "platforms", "scopes", "importance"}


class CatalogError(ValueError):
    """The catalog (or an overlay) is malformed or violates a catalog rule."""


@dataclass(frozen=True)
class Scope:
    """What gets monitored: one source, one platform, selected by tag and name globs."""

    name: str
    source: str
    platform: str
    tags: tuple[str, ...] = ()
    benchmarks: tuple[str, ...] = ("*", )
    groups: tuple[str, ...] = ()
    expected_nightly: bool = False


@dataclass(frozen=True)
class Catalog:  # pylint: disable=too-many-instance-attributes
    """Parsed catalog. Build with :func:`load_catalog`."""

    defaults: Mapping[str, float]
    metrics: Mapping[str, bool]  # metric -> higher_is_better
    platforms: Mapping[str, Mapping[str, Any]]
    families: Mapping[str, Mapping[str, Any]]
    benchmarks: Mapping[str, Mapping[str, Any]]
    scopes: Mapping[str, Scope] = field(default_factory=dict)
    importance: Mapping[str, Any] = field(default_factory=dict)
    ungated_metrics: frozenset[str] = frozenset()  # declared, but never gated (`gate: false`)

    def normalize_platform(self, gpu_device: str) -> str | None:
        """Map a raw gpu_device string to a platform label, or None if no platform matches."""
        for label, spec in self.platforms.items():
            if re.search(spec["match"], gpu_device):
                return label
        return None

    def _first_family(self, benchmark: str) -> tuple[str, Mapping[str, Any]] | None:
        """First family (in declaration order) whose glob matches *benchmark*."""
        for name, spec in self.families.items():
            if any(fnmatch.fnmatchcase(benchmark, pat) for pat in spec["match"]):
                return name, spec
        return None

    def family_of(self, benchmark: str) -> str:
        """Name of the first matching family in declaration order, else the benchmark name itself."""
        hit = self._first_family(benchmark)
        return benchmark if hit is None else hit[0]

    def metric_state(self, metric: str) -> str:
        """``"gated"``, ``"ungated"`` (declared with ``gate: false``) or ``"undeclared"``."""
        if metric not in self.metrics:
            return "undeclared"
        return "ungated" if metric in self.ungated_metrics else "gated"

    def max_window(self) -> int:
        """Largest ``window`` configured anywhere: how many nights of history a gate may need."""
        specs = [self.defaults, *self.families.values(), *self.benchmarks.values(), *self.platforms.values()]
        return max(int(spec["window"]) for spec in specs if "window" in spec)

    def resolve(self, key: SeriesKey) -> Thresholds:
        """Thresholds for *key*: defaults -> first matching family -> benchmark -> platform.

        Families are matched in declaration order and only the FIRST match applies, so list the
        more specific family first. An undeclared metric raises: its direction is unknown.
        """
        if key.metric not in self.metrics:
            raise CatalogError(f"metric {key.metric!r} is not declared in the catalog")
        merged: dict[str, Any] = dict(self.defaults)
        hit = self._first_family(key.benchmark)
        if hit is not None:
            merged.update({k: v for k, v in hit[1].items() if k in _THRESHOLD_KEYS})
        merged.update(self.benchmarks.get(key.benchmark, {}))
        merged.update({k: v for k, v in self.platforms.get(key.platform, {}).items() if k in _THRESHOLD_KEYS})
        return Thresholds(higher_is_better=self.metrics[key.metric], **merged)

    def validate(self, known_benchmarks: Iterable[str] | None = None, strict: bool = False) -> None:
        """Raise CatalogError if the catalog is inconsistent.

        With ``strict`` and *known_benchmarks*, every scope/benchmark selector and at least one
        pattern per family must match at least one known benchmark name.
        """
        for scope in self.scopes.values():
            if scope.platform not in self.platforms:
                raise CatalogError(f"scope {scope.name!r}: unknown platform {scope.platform!r}")
        patterns = [(n, p) for n, spec in self.families.items() for p in spec["match"]]
        for i, (fam, pat) in enumerate(patterns):
            for other, opat in patterns[i + 1:]:
                if fam != other and (fnmatch.fnmatchcase(pat, opat) or fnmatch.fnmatchcase(opat, pat)):
                    warnings.warn(
                        f"families {fam!r} and {other!r} overlap ({pat!r} vs {opat!r}); "
                        f"{fam!r} wins for benchmarks matching both", stacklevel=2)
        if not (strict and known_benchmarks is not None):
            return
        known = list(known_benchmarks)
        selectors = [(f"scope {s.name!r}", p) for s in self.scopes.values() for p in s.benchmarks]
        selectors += [(f"benchmark {n!r}", n) for n in self.benchmarks]
        for where, pat in selectors:
            if not any(fnmatch.fnmatchcase(b, pat) for b in known):
                raise CatalogError(f"{where}: selector {pat!r} matches no known benchmark")
        for name, spec in self.families.items():  # a family needs only one live pattern
            if not any(fnmatch.fnmatchcase(b, pat) for b in known for pat in spec["match"]):
                raise CatalogError(f"family {name!r}: no pattern matches a known benchmark")


def _check_thresholds(where: str, spec: Mapping[str, Any],
                      extra: set[str] = frozenset()) -> None:  # type: ignore[assignment]
    unknown = set(spec) - _THRESHOLD_KEYS - extra
    if unknown:
        raise CatalogError(f"{where}: unknown keys {sorted(unknown)}")
    for k in _THRESHOLD_KEYS & set(spec):
        v = spec[k]
        integer, low, exclusive = _RULES[k]
        ok = isinstance(v, int) if integer else isinstance(v, (int, float))
        ok = ok and not isinstance(v, bool) and math.isfinite(v) and (v > low if exclusive else v >= low)
        if not ok:
            kind = "an integer" if integer else "a finite number"
            bound = (">" if exclusive else ">=") + f" {low}"
            raise CatalogError(f"{where}: {k} must be {kind} {bound}, got {v!r}")


def _as_mapping(where: str, value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CatalogError(f"{where}: expected a mapping" +
                           (" (is the file path correct?)" if where == "catalog" else ""))
    return value


def _parse(raw: Any) -> dict[str, Any]:
    raw = _as_mapping("catalog", raw)
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise CatalogError(f"catalog: unknown keys {sorted(unknown)}")
    if raw.get("version") != 1:
        raise CatalogError("catalog: unsupported version " + repr(raw.get("version")))
    return dict(raw)


def _load_yaml(source: str | Path) -> dict[str, Any]:
    """Read a catalog from an existing path, otherwise parse *source* as YAML text."""
    is_path = isinstance(source, Path) or ("\n" not in source and Path(source).is_file())
    try:
        text = Path(source).read_text(encoding="utf-8") if is_path else str(source)
        return _parse(yaml.safe_load(text))
    except yaml.YAMLError as exc:
        raise CatalogError(f"invalid YAML: {exc}") from exc


def _parse_scopes(raw: Mapping[str, Any]) -> dict[str, Scope]:
    scopes = {}
    for name, spec in raw.items():
        spec = _as_mapping(f"scope {name!r}", spec)
        unknown = set(spec) - _SCOPE_KEYS
        if unknown or not {"source", "platform"} <= set(spec):
            raise CatalogError(f"scope {name!r}: needs source+platform; unknown keys {sorted(unknown)}")
        if spec["source"] not in _SOURCE_KINDS:
            got = spec["source"]
            raise CatalogError(f"scope {name!r}: source must be one of {sorted(_SOURCE_KINDS)}, got {got!r}")
        for field_name in ("tags", "benchmarks", "groups"):
            value = spec.get(field_name, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                raise CatalogError(f"scope {name!r}: {field_name} must be a list of strings")
        kwargs = {k: tuple(v) if isinstance(v, list) else v for k, v in spec.items()}
        scopes[name] = Scope(name=name, **kwargs)
    return scopes


def _build(raw: Mapping[str, Any]) -> Catalog:
    defaults = _as_mapping("defaults", raw.get("defaults"))
    _check_thresholds("defaults", defaults)
    missing = _REQUIRED_KEYS - set(defaults)
    if missing:
        raise CatalogError(f"defaults: missing {sorted(missing)}")
    metrics, ungated = {}, set()
    for name, spec in _as_mapping("metrics", raw.get("metrics")).items():
        spec = _as_mapping(f"metric {name!r}", spec)
        if "higher_is_better" not in spec or set(spec) - _METRIC_KEYS or not all(
                isinstance(v, bool) for v in spec.values()):
            raise CatalogError(f"metric {name!r}: expected boolean higher_is_better and optional boolean gate")
        metrics[name] = spec["higher_is_better"]
        if spec.get("gate") is False:
            ungated.add(name)
    platforms = _as_mapping("platforms", raw.get("platforms"))
    for name, spec in platforms.items():
        _check_thresholds(f"platform {name!r}", spec, {"match"})
        try:
            re.compile(spec["match"])
        except (KeyError, TypeError, re.error) as exc:
            raise CatalogError(f"platform {name!r}: needs a valid regex 'match'") from exc
    families = _as_mapping("families", raw.get("families"))
    for name, spec in families.items():
        _check_thresholds(f"family {name!r}", spec, {"match"})
        if not isinstance(spec.get("match"), list) or not spec["match"]:
            raise CatalogError(f"family {name!r}: 'match' must be a non-empty list of globs")
    benchmarks = _as_mapping("benchmarks", raw.get("benchmarks"))
    for name, spec in benchmarks.items():
        _check_thresholds(f"benchmark {name!r}", spec)
    catalog = Catalog(defaults, metrics, platforms, families, benchmarks,
                      _parse_scopes(_as_mapping("scopes", raw.get("scopes"))),
                      _as_mapping("importance", raw.get("importance")), frozenset(ungated))
    catalog.validate()
    return catalog


def load_catalog(path_or_text: str | Path | None = None, overlays: Sequence[str | Path] = ()) -> Catalog:
    """Load a catalog from a path or YAML text (default: the shipped default.yaml).

    *overlays* may add platforms, scopes and importance; they may not change thresholds
    (defaults, metrics, families, benchmarks, or threshold keys on a platform), nor redefine anything.
    """
    raw = _load_yaml(DEFAULT_CATALOG_PATH if path_or_text is None else path_or_text)
    for overlay in overlays:
        extra = _load_yaml(overlay)
        forbidden = set(extra) - _OVERLAY_KEYS
        if forbidden:
            raise CatalogError(f"overlay may not change {sorted(forbidden)}")
        for section in ("platforms", "scopes", "importance"):
            add = _as_mapping(f"overlay {section}", extra.get(section))
            if section == "platforms":
                for name, spec in add.items():
                    if set(spec) - {"match"}:
                        raise CatalogError(f"overlay platform {name!r}: may only set 'match'")
            clash = set(add) & set(_as_mapping(section, raw.get(section)))
            if clash:
                raise CatalogError(f"overlay redefines {section}: {sorted(clash)}")
            raw[section] = {**_as_mapping(section, raw.get(section)), **add}
    return _build(raw)


__all__ = ["Catalog", "CatalogError", "Scope", "load_catalog", "DEFAULT_CATALOG_PATH", "SCHEMA_PATH"]
