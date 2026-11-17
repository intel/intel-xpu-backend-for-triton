"""Regression gate: modified-z + relative-drop dual gate with improvement lock-in.

Ports ``analyze()`` from #6491 to the ``Series``/``Thresholds`` model. #6491 works per run; here each
``Point`` plays the role of one run (aggregation is not this module's job).

Percentages are relative to the magnitude of the baseline (``abs(median)``). Metrics such as ``ns_per_arg``
are differences of two timings and can be negative; dividing by a negative median would flip the sign of
the relative change against the sign of the z-score, so no verdict could ever fire.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from datetime import date

from benchmark_monitor.catalog import Catalog
from benchmark_monitor.model import Point, Series, Status, Thresholds, Verdict

_MAD_SCALE = 0.6745
_MAD_PROXY_FRAC = 0.01


def _mad(values: list[float], center: float) -> float:
    return statistics.median([abs(v - center) for v in values])


def _baseline(history: list[float], thresholds: Thresholds, sign: float) -> tuple[float, float]:
    """Baseline median and MAD, locked to the recent level after a stable improvement."""
    full_median = statistics.median(history)
    recent = history[-max(thresholds.min_history, min(int(thresholds.lock_window), len(history))):]
    recent_median = statistics.median(recent)
    recent_mad = _mad(recent, recent_median)
    gain_pct = sign * (recent_median - full_median) / abs(full_median) * 100 if full_median else 0.0
    if gain_pct > thresholds.improvement_lock_pct and recent_mad < abs(recent_median) * thresholds.lock_mad_frac:
        median = recent_median
        mad = recent_mad if recent_mad > 0 else abs(recent_median) * _MAD_PROXY_FRAC
    else:
        median = full_median
        mad = _mad(history, full_median)
    if mad == 0:  # 1% of the baseline as a MAD proxy
        mad = abs(median) * _MAD_PROXY_FRAC if median != 0 else 1e-6
    return median, mad


def evaluate(series: Series, thresholds: Thresholds, night: date | None = None) -> Verdict:
    """Verdict for *series* on *night* (default: its last point).

    The baseline is the previous ``window`` points; a night without a point is skipped. With unit="run",
    tonight is the night's last point and earlier runs stay in the baseline.
    """
    if night is None:
        if not series.points:
            raise ValueError("cannot evaluate an empty series without an explicit night")
        night = series.points[-1].night
    idx = max((i for i, p in enumerate(series.points) if p.night == night), default=None)
    current: Point | None = None if idx is None else series.points[idx]
    history = [p.value for p in series.points[:idx]][-thresholds.window:] if idx is not None else []

    def verdict(status: Status, **kw: float | None) -> Verdict:
        n_history = 0 if status is Status.MISSING else len(history)
        return Verdict(series.key, night, status, kw.get("drop_pct"), kw.get("z"), kw.get("median"), kw.get("mad"),
                       n_history, thresholds)

    if current is None:
        return verdict(Status.MISSING)
    if len(history) < thresholds.min_history:
        return verdict(Status.INSUFFICIENT_HISTORY)

    # CV: sample stdev over the whole window relative to |mean|, only when the mean is not zero.
    mean = statistics.mean(history)
    if mean != 0 and len(history) >= 2 and statistics.stdev(history) / abs(mean) * 100 > thresholds.max_cv_pct:
        return verdict(Status.NOISY)

    sign = 1.0 if thresholds.higher_is_better else -1.0
    median, mad = _baseline(history, thresholds, sign)

    delta = sign * (current.value - median)
    z = _MAD_SCALE * delta / mad
    if median == 0:
        if current.value == 0:
            return verdict(Status.OK, drop_pct=0.0, z=0.0, median=median, mad=mad)
        change_pct = delta / 1e-6 * 100
    else:
        change_pct = delta / abs(median) * 100
    drop_pct = -change_pct  # positive = worse
    if z < -thresholds.z and change_pct < -thresholds.min_drop_pct:
        status = Status.REGRESSION
    elif z > thresholds.z and change_pct > thresholds.min_drop_pct:
        status = Status.IMPROVEMENT
    else:
        status = Status.OK
    return verdict(status, drop_pct=drop_pct, z=z, median=median, mad=mad)


def evaluate_all(series_iter: Iterable[Series], catalog: Catalog) -> list[Verdict]:
    """Gate every non-empty series on its last night with catalog-resolved thresholds.

    Series of metrics declared with ``gate: false`` are skipped; an undeclared metric raises CatalogError.
    """
    return [evaluate(s, catalog.resolve(s.key)) for s in series_iter if s.points and catalog.is_gated(s.key.metric)]
