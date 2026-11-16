"""Collapse raw observations of one series into one point per night (or per run)."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime, timezone
from typing import Literal

from benchmark_monitor.model import Observation, Point, Series


def _utc(moment: datetime) -> datetime:
    """Aware UTC form of *moment*; naive datetimes are taken as UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def utc_night(moment: datetime) -> date:
    """UTC calendar date of *moment*."""
    return _utc(moment).date()


def aggregate(observations: Iterable[Observation], unit: Literal["night_mean", "run"] = "night_mean") -> Series:
    """Aggregate observations sharing one SeriesKey.

    ``night_mean``: one point per UTC night, value = mean of that night's observations.
    ``run``: one point per run_id (mean over duplicate rows of the run), kept in time order.
    """
    if unit not in ("night_mean", "run"):
        raise ValueError(f"unknown aggregation unit: {unit!r}")
    obs = list(observations)
    if not obs:
        raise ValueError("cannot aggregate zero observations")
    key = obs[0].key
    if any(o.key != key for o in obs):
        raise ValueError("observations belong to more than one SeriesKey")

    groups: dict[object, list[Observation]] = defaultdict(list)
    for o in sorted(obs, key=lambda o: (_utc(o.observed_at), o.run_id)):
        groups[utc_night(o.observed_at) if unit == "night_mean" else o.run_id].append(o)

    points = [
        Point(night=utc_night(g[0].observed_at), value=statistics.fmean(o.value
                                                                        for o in g), n_raw=len(g),
              run_ids=tuple(sorted({o.run_id
                                    for o in g})))
        for g in groups.values()
    ]
    points.sort(key=lambda p: p.night)  # stable: runs on one night stay in time order
    return Series(key=key, points=tuple(points))
