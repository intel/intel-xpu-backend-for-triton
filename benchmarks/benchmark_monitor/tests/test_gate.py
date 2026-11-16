"""Gate tests, ported from #6491 test_analysis.py (TestAnalyze, lock-in, MAD=0, zero baseline, CV)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from benchmark_monitor import Point, Series, SeriesKey, Status, Thresholds, load_catalog
from benchmark_monitor.gate import evaluate, evaluate_all

KEY = SeriesKey("micro", "b580", "ci", "gemm", "triton", "tflops", "{}")
# #6491 DEFAULT_CONFIG: min_history=5, rolling_window=10, z=3, drop=5, max_cv=15%, lock 5%.
TH = Thresholds(5.0, 3.0, 5, 10, 15.0, 0.05, True)
TH20 = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True)


def mk(values: list[float], key: SeriesKey = KEY) -> Series:
    start = date(2025, 1, 1)
    return Series(key, tuple(Point(start + timedelta(i), v, 1, (f"r{i}", )) for i, v in enumerate(values)))


def status(values: list[float], th: Thresholds = TH) -> Status:
    return evaluate(mk(values), th).status


def test_empty_series_needs_night():
    with pytest.raises(ValueError):
        evaluate(mk([]), TH)


def test_stable_ok():
    v = evaluate(mk([10.0] * 10), TH)
    assert v.status is Status.OK and v.n_history == 9


def test_clear_regression():
    v = evaluate(mk([10.0] * 10 + [8.0]), TH)
    assert v.status is Status.REGRESSION
    assert v.baseline_median == 10.0 and v.drop_pct == pytest.approx(20.0)


def test_clear_improvement():
    v = evaluate(mk([10.0] * 10 + [12.0]), TH)
    assert v.status is Status.IMPROVEMENT and v.drop_pct == pytest.approx(-20.0)


def test_insufficient_history():
    assert status([10.0] * 3 + [5.0]) is Status.INSUFFICIENT_HISTORY


def test_mad_zero_proxy_is_one_percent_of_median():
    v = evaluate(mk([10.0] * 10 + [8.0]), TH)
    assert v.mad == pytest.approx(0.1)
    assert v.z == pytest.approx(0.6745 * -2.0 / 0.1)


def test_small_drop_not_flagged_with_proxy_mad():
    assert status([10.0] * 10 + [9.7]) is Status.OK


def test_z_ok_but_drop_below_min_drop():
    big = Thresholds(25.0, 3.0, 5, 10, 15.0, 0.05, True)
    assert status([10.0] * 10 + [8.0], big) is Status.OK


def test_window_limits_baseline():
    v = evaluate(mk([1.0] * 10 + [10.0] * 10 + [8.0]), TH)
    assert v.n_history == 10 and v.baseline_median == 10.0 and v.status is Status.REGRESSION


def test_night_selects_point_and_excludes_later():
    s = mk([10.0] * 8 + [8.0, 99.0])
    v = evaluate(s, TH, night=date(2025, 1, 9))
    assert v.status is Status.REGRESSION and v.n_history == 8


def test_missing_night():
    assert evaluate(mk([10.0] * 8), TH, night=date(2025, 3, 1)).status is Status.MISSING


def test_lock_in_stable_improvement():
    v = evaluate(mk([100.0] * 15 + [120.0] * 5 + [105.0]), TH20)
    assert v.status is Status.REGRESSION and v.baseline_median == 120.0


def test_lock_in_stable_improvement_8_recent():
    v = evaluate(mk([100.0] * 12 + [120.0] * 8 + [105.0]), TH20)
    assert v.status is Status.REGRESSION and v.baseline_median == 120.0


def test_no_lock_in_small_gain():
    assert status([100.0] * 15 + [105.0] * 5 + [100.0], TH20) is Status.OK


def test_noisy_improvement_no_lock_in():
    vals = [100.0] * 12 + [130, 90, 135, 85, 140, 80, 130, 95] + [100.0]
    assert evaluate(mk(vals), TH20).status is Status.NOISY  # #6491 skips on CV before lock-in is reached
    v = evaluate(mk(vals), Thresholds(5.0, 3.0, 5, 20, 100.0, 0.05, True))
    assert v.status is Status.OK and v.baseline_median == 100.0


def test_lock_mad_frac_is_configurable():
    vals = [100.0] * 12 + [118, 122, 118, 122, 118, 122, 118, 122] + [105.0]
    strict = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.01, True)
    assert evaluate(mk(vals), strict).baseline_median == 100.0
    assert evaluate(mk(vals), TH20).baseline_median == 120.0


def test_zero_baseline():
    assert status([0.0] * 10 + [0.0]) is Status.OK
    assert status([0.0] * 10 + [10.0]) is Status.IMPROVEMENT


def test_bimodal_is_noisy():
    assert status([17.0, 29.0] * 5 + [17.0]) is Status.NOISY
    loose = Thresholds(5.0, 3.0, 5, 10, 50.0, 0.05, True)
    assert status([17.0, 29.0] * 5 + [17.0], loose) is not Status.NOISY


def test_moderate_noise_regresses():
    vals = [90.0, 95.0, 100.0, 105.0, 110.0, 92.0, 98.0, 102.0, 108.0, 96.0, 75.0]
    assert status(vals) is Status.REGRESSION


def test_lower_is_better_mirrors():
    lo = Thresholds(5.0, 3.0, 5, 10, 15.0, 0.05, False)
    assert status([10.0] * 10 + [12.0], lo) is Status.REGRESSION
    assert status([10.0] * 10 + [8.0], lo) is Status.IMPROVEMENT
    v = evaluate(mk([100.0] * 15 + [80.0] * 5 + [95.0]), Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, False))
    assert v.status is Status.REGRESSION and v.baseline_median == 80.0


def test_evaluate_all_uses_catalog():
    cat = load_catalog()
    out = evaluate_all([mk([10.0] * 10 + [5.0]), mk([])], cat)
    assert len(out) == 1 and out[0].thresholds == cat.resolve(KEY)
    assert out[0].status is Status.REGRESSION
