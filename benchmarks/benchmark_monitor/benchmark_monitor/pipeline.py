"""observations -> aggregate -> per-night gate -> suppression -> Report."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import astuple, replace
from datetime import date, timedelta

from benchmark_monitor.aggregate import aggregate
from benchmark_monitor.catalog import Catalog
from benchmark_monitor.gate import evaluate
from benchmark_monitor.model import Observation, Series, SeriesKey, Thresholds, Verdict
from benchmark_monitor.report import Report
from benchmark_monitor.source import Source, gated_observations
from benchmark_monitor.suppress import SuppressConfig, apply, identity

Gate = Callable[[Series, Thresholds, date], Verdict]


def run(catalog: Catalog, source: Source, scope_names: Iterable[str], as_of: date, nights: int,
        suppress_cfg: SuppressConfig | None = None, gate: Gate | None = None) -> Report:
    """Gate and suppress the last *nights* nights up to *as_of* (inclusive).

    The source is read from ``catalog.max_window()`` nights before the first evaluated night, so the
    first night has a baseline too (one point per night is assumed). ``gate`` defaults to
    ``benchmark_monitor.gate.evaluate``; tests may inject a fake.
    """
    if nights < 1:
        raise ValueError(f"nights must be at least 1, got {nights}")
    gate = gate or evaluate
    cfg = suppress_cfg or SuppressConfig()
    days = [as_of - timedelta(days=i) for i in range(nights - 1, -1, -1)]
    start = days[0] - timedelta(days=catalog.max_window())
    grouped: dict[SeriesKey, list[Observation]] = defaultdict(list)
    warnings: list[str] = []
    if cfg.flip_episodes > 0 and nights < cfg.flip_days:
        warnings.append(f"nights={nights} is shorter than the flip-flop look-back of {cfg.flip_days}; "
                        "flip-flop muting may miss earlier episodes")
    for name in scope_names:
        raw = list(source.observations(catalog.scopes[name], start, as_of))
        if not raw:
            warnings.append(f"scope {name}: no observations in {start}..{as_of}")
        kept, skipped = gated_observations(catalog, raw)
        warnings += [w for w in skipped if w not in warnings]
        for obs in kept:
            grouped[obs.key].append(obs)
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
