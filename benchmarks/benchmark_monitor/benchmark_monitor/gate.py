"""Regression gate: modified-z + relative-drop dual gate with improvement lock-in.

Ports ``analyze()`` from intel-xpu-backend-for-triton #6491 (``analysis.py``) to the
``Series``/``Thresholds`` model. Line references below are to that file. #6491 works
per run; here each ``Point`` plays the role of one run (aggregation is not this module's job).
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from datetime import date

from benchmark_monitor.catalog import Catalog
from benchmark_monitor.model import Point, Series, Status, Thresholds, Verdict

# #6491 thresholds.yaml `improvement_lock_pct` default; Thresholds has no field for it.
IMPROVEMENT_LOCK_PCT = 8.0
# #6491 analysis.py:119 `min(8, ...)`: cap on the "recent" lock-in window.
LOCK_WINDOW_CAP = 8
_MAD_SCALE = 0.6745  # analysis.py:141
_MAD_PROXY_FRAC = 0.01  # analysis.py:132,138


def _mad(values: list[float], center: float) -> float:
    return statistics.median([abs(v - center) for v in values])


def _baseline(history: list[float], thresholds: Thresholds, sign: float) -> tuple[float, float]:
    """Baseline median and MAD, locked to the recent level after a stable improvement."""
    full_median = statistics.median(history)
    # analysis.py:118-127
    recent = history[-max(thresholds.min_history, min(LOCK_WINDOW_CAP, len(history))):]
    recent_median = statistics.median(recent)
    recent_mad = _mad(recent, recent_median)
    gain_pct = sign * (recent_median - full_median) / full_median * 100 if full_median else 0.0
    if gain_pct > IMPROVEMENT_LOCK_PCT and recent_mad < recent_median * thresholds.lock_mad_frac:
        median = recent_median
        mad = recent_mad if recent_mad > 0 else recent_median * _MAD_PROXY_FRAC  # analysis.py:132
    else:
        median = full_median
        mad = _mad(history, full_median)
    if mad == 0:  # analysis.py:136-139: 1% of baseline as MAD proxy
        mad = median * _MAD_PROXY_FRAC if median > 0 else 1e-6
    return median, mad


def evaluate(series: Series, thresholds: Thresholds, night: date | None = None) -> Verdict:
    """Gate ``series`` on ``night`` (default: its last point) against the preceding points."""
    if night is None:
        if not series.points:
            raise ValueError("cannot evaluate an empty series without an explicit night")
        night = series.points[-1].night
    current: Point | None = next((p for p in series.points if p.night == night), None)
    # analysis.py:101: baseline = last `window` points strictly before tonight.
    history = [p.value for p in series.points if p.night < night][-thresholds.window:]

    def verdict(status: Status, **kw: float | None) -> Verdict:
        return Verdict(series.key, night, status, kw.get("drop_pct"), kw.get("z"), kw.get("median"), kw.get("mad"),
                       len(history), thresholds)

    if current is None:
        return verdict(Status.MISSING)
    if len(history) < thresholds.min_history:  # analysis.py:104
        return verdict(Status.INSUFFICIENT_HISTORY)

    # analysis.py:109-114: sample-stdev CV over the whole window, only when mean > 0.
    mean = statistics.mean(history)
    if mean > 0 and len(history) >= 2 and statistics.stdev(history) / mean * 100 > thresholds.max_cv_pct:
        return verdict(Status.NOISY)

    sign = 1.0 if thresholds.higher_is_better else -1.0
    median, mad = _baseline(history, thresholds, sign)

    delta = sign * (current.value - median)
    z = _MAD_SCALE * delta / mad  # analysis.py:141
    if median == 0:  # analysis.py:143-151
        if current.value == 0:
            return verdict(Status.OK, drop_pct=0.0, z=0.0, median=median, mad=mad)
        change_pct = delta / 1e-6 * 100
    else:
        change_pct = delta / median * 100
    drop_pct = -change_pct  # positive = worse
    if z < -thresholds.z and change_pct < -thresholds.min_drop_pct:  # analysis.py:155
        status = Status.REGRESSION
    elif z > thresholds.z and change_pct > thresholds.min_drop_pct:  # analysis.py:156
        status = Status.IMPROVEMENT
    else:
        status = Status.OK
    return verdict(status, drop_pct=drop_pct, z=z, median=median, mad=mad)


def evaluate_all(series_iter: Iterable[Series], catalog: Catalog) -> list[Verdict]:
    """Gate every non-empty series on its last night with catalog-resolved thresholds."""
    return [evaluate(s, catalog.resolve(s.key)) for s in series_iter if s.points]
