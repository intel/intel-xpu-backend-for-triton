"""Frozen domain model shared by every benchmark_monitor layer (stdlib only)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class Status(str, Enum):
    """Outcome of gating one series on one night."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    OK = "ok"
    INSUFFICIENT_HISTORY = "insufficient_history"
    NOISY = "noisy"
    MISSING = "missing"


def canonical_params(params: Mapping[str, object]) -> str:
    """Canonical JSON of shape params: sorted keys, no spaces."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SeriesKey:
    """Identity of one monitored time series."""

    source: str  # "micro" | "third_party" | "artifact" (free text allowed)
    platform: str  # normalized label from catalog.platforms, e.g. "b580"
    tag: str
    benchmark: str
    compiler: str
    metric: str
    params: str  # canonical JSON, see canonical_params()


@dataclass(frozen=True)
class Observation:
    """One raw measurement. ``attrs`` holds versions and provenance; empty strings are omitted."""

    key: SeriesKey
    run_id: str
    observed_at: datetime
    value: float
    attrs: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Point:
    """One aggregated data point (one per night, or one per run)."""

    night: date
    value: float
    n_raw: int
    run_ids: tuple[str, ...]


@dataclass(frozen=True)
class Series:
    """Aggregated points for one key, sorted by night ascending."""

    key: SeriesKey
    points: tuple[Point, ...]


@dataclass(frozen=True)
class Thresholds:  # pylint: disable=too-many-instance-attributes
    """Resolved gate thresholds for one series."""

    min_drop_pct: float
    z: float
    min_history: int
    window: int
    max_cv_pct: float
    lock_mad_frac: float
    higher_is_better: bool
    improvement_lock_pct: float = 8.0  # lock the baseline when the recent level is this % better
    lock_window: int = 8  # cap on the number of recent points used for the lock-in level


@dataclass(frozen=True)
class Verdict:  # pylint: disable=too-many-instance-attributes
    """Gate result for one series on one night."""

    key: SeriesKey
    night: date
    status: Status
    drop_pct: float | None
    z: float | None
    baseline_median: float | None
    mad: float | None
    n_history: int
    thresholds: Thresholds
