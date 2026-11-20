"""observations -> aggregate -> per-night gate -> suppression -> Report."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import astuple, replace
from datetime import date, datetime, timedelta

from benchmark_monitor.aggregate import aggregate
from benchmark_monitor.catalog import Catalog
from benchmark_monitor.gate import evaluate
from benchmark_monitor.model import Observation, Series, SeriesKey, Thresholds, Verdict
from benchmark_monitor.report import Report
from benchmark_monitor.source import Source, gated_observations
from benchmark_monitor.suppress import SuppressConfig, apply, identity


def history_start(as_of: date, history_days: int | None, max_window: int) -> date:
    """First night to read. The gate's window counts points, not days, so a series with gaps needs a wider
    calendar span than ``max_window``: default 3x, 0 = unbounded. The gate trims to its last ``window`` points."""
    if history_days is None:
        history_days = max_window * 3
    if history_days <= 0:
        return date.min
    return as_of - timedelta(days=history_days)


Gate = Callable[[Series, Thresholds, date], Verdict]


def run(catalog: Catalog, source: Source, scope_names: Iterable[str], as_of: date, nights: int,
        suppress_cfg: SuppressConfig | None = None, gate: Gate | None = None,
        history_days: int | None = None) -> Report:
    """Gate and suppress the last *nights* nights up to *as_of* (inclusive).

    The source is read from ``history_days`` calendar days before the first evaluated night (default 3 x
    ``catalog.max_window()``, see ``history_start``), so the first night has a baseline even when some nights
    have no run; the gate takes the last ``window`` points. Observations selected by several scopes are
    deduplicated by (series key, run id, observed_at), so overlapping scopes never double-count.
    ``gate`` defaults to ``benchmark_monitor.gate.evaluate``; tests may inject a fake.
    """
    if nights < 1:
        raise ValueError(f"nights must be at least 1, got {nights}")
    gate = gate or evaluate
    cfg = suppress_cfg or SuppressConfig()
    days = [as_of - timedelta(days=i) for i in range(nights - 1, -1, -1)]
    start = history_start(days[0], history_days, catalog.max_window())
    grouped: dict[SeriesKey, list[Observation]] = defaultdict(list)
    warnings: dict[str, None] = {}  # ordered set
    if cfg.flip_episodes > 0 and nights < cfg.flip_days:
        warnings[f"nights={nights} is shorter than the flip-flop look-back of {cfg.flip_days}; "
                 "flip-flop muting may miss earlier episodes"] = None
    if nights < cfg.confirm_nights:
        warnings[f"nights={nights} is below confirm_nights={cfg.confirm_nights}: nothing can be confirmed, "
                 "so a standing regression is reported as new every night"] = None
    seen: set[tuple[SeriesKey, str, datetime]] = set()
    for name in dict.fromkeys(scope_names):  # a repeated scope is read once
        raw = list(source.observations(catalog.scopes[name], start, as_of))
        if not raw:
            warnings[f"scope {name}: no observations in {start}..{as_of}"] = None
        kept, skipped = gated_observations(catalog, raw)
        warnings.update(dict.fromkeys(skipped))
        this_scope: set[tuple[SeriesKey, str, datetime]] = set()
        for obs in kept:
            ident = (obs.key, obs.run_id, obs.observed_at)
            if ident not in seen:  # drop only what an earlier scope returned; rows within one scope all count
                this_scope.add(ident)
                grouped[obs.key].append(obs)
        seen |= this_scope
    history: dict[date, list[Verdict]] = {d: [] for d in days}
    for key, obs in sorted(grouped.items(), key=lambda kv: astuple(kv[0])):
        series, thresholds = aggregate(obs), catalog.resolve(key)
        for d in days:
            history[d].append(gate(series, thresholds, d))
    if cfg.family_of is identity:  # a custom family_of wins over the catalog's
        cfg = replace(cfg, family_of=catalog.family_of)
    result = apply(history, cfg, as_of)
    return Report(as_of=as_of, verdicts=tuple(history[as_of]), suppression=result,
                  warnings=tuple(warnings) + result.warnings)
