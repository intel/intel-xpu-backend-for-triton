"""Gate tests, ported from #6491 test_analysis.py (TestAnalyze, lock-in, MAD=0, zero baseline, CV)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from benchmark_monitor import CatalogError, Point, Series, SeriesKey, Status, Thresholds, load_catalog
from benchmark_monitor.gate import evaluate, evaluate_all

KEY = SeriesKey("micro", "b580", "ci", "gemm", "triton", "tflops", "{}")
# #6491 DEFAULT_CONFIG: min_history=5, rolling_window=10, z=3, drop=5, max_cv=15%, lock_mad_frac=0.05.
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
    v = evaluate(mk([10.0] * 8), TH, night=date(2025, 3, 1))
    assert v.status is Status.MISSING and v.n_history == 0


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


LOWER = Thresholds(5.0, 3.0, 5, 10, 15.0, 0.05, False)


def test_negative_baseline_lower_is_better_regresses():
    # ns_per_arg can be negative: -8 is a higher (worse) value than -10 for a lower-is-better metric.
    v = evaluate(mk([-10.0] * 10 + [-8.0]), LOWER)
    assert v.status is Status.REGRESSION
    assert v.drop_pct == pytest.approx(20.0) and v.z is not None and v.z < 0
    assert v.baseline_median == -10.0 and v.mad == pytest.approx(0.1)  # MAD proxy stays positive
    assert evaluate(mk([-10.0] * 10 + [-12.0]), LOWER).status is Status.IMPROVEMENT


def test_negative_baseline_higher_is_better_mirrors():
    assert status([-10.0] * 10 + [-12.0]) is Status.REGRESSION
    assert status([-10.0] * 10 + [-8.0]) is Status.IMPROVEMENT
    assert status([-10.0] * 10 + [-9.9]) is Status.OK


def test_negative_history_is_still_filtered_for_noise():
    assert status([-17.0, -29.0] * 5 + [-17.0]) is Status.NOISY


def test_negative_baseline_lock_in():
    th = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, False)
    v = evaluate(mk([-100.0] * 15 + [-120.0] * 5 + [-105.0]), th)  # more negative = lower = better: locks at -120
    assert v.baseline_median == -120.0 and v.status is Status.REGRESSION


def test_evaluate_all_uses_catalog():
    cat = load_catalog()
    out = evaluate_all([mk([10.0] * 10 + [5.0]), mk([])], cat)
    assert len(out) == 1 and out[0].thresholds == cat.resolve(KEY)
    assert out[0].status is Status.REGRESSION


def test_evaluate_all_skips_ungated_metrics_and_rejects_undeclared():
    cat = load_catalog()
    ungated = SeriesKey("micro", "b580", "ci", "gemm", "triton", "cpu_mhz", "{}")
    out = evaluate_all([mk([10.0] * 10 + [5.0], ungated), mk([10.0] * 10 + [5.0])], cat)
    assert [v.key.metric for v in out] == ["tflops"]
    with pytest.raises(CatalogError, match="not declared"):
        evaluate_all([mk([1.0], SeriesKey("micro", "b580", "ci", "gemm", "triton", "nope", "{}"))], cat)


def test_cv_exactly_at_max_cv_is_not_noisy():
    # mean 100, sample stdev exactly 15.0 -> CV is exactly 15%; the gate skips only when CV > max_cv_pct.
    hist = [85.0, 115.0, 85.0, 115.0, 100.0]
    assert evaluate(mk(hist + [100.0]), TH).status is Status.OK
    assert evaluate(mk(hist + [100.0]), Thresholds(5.0, 3.0, 5, 10, 14.99, 0.05, True)).status is Status.NOISY


def test_baseline_is_previous_window_points_not_calendar_nights():
    start = date(2025, 1, 1)
    # 12 points with a 2-night gap after the 6th; window=10 keeps the last 10 points before tonight.
    days = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]
    vals = [1.0] + [10.0] * 10 + [8.0]
    s = Series(KEY, tuple(Point(start + timedelta(d), v, 1, (f"r{d}", )) for d, v in zip(days, vals)))
    v = evaluate(s, TH)
    assert v.n_history == 10 and v.baseline_median == 10.0 and v.status is Status.REGRESSION


def test_night_without_point_is_skipped_not_a_gap():
    start = date(2025, 1, 1)
    days = [0, 1, 2, 3, 4, 5, 6, 8, 9]  # night 7 has no point
    s = Series(KEY,
               tuple(Point(start + timedelta(d), 10.0 if i < 8 else 8.0, 1, (f"r{d}", )) for i, d in enumerate(days)))
    v = evaluate(s, TH)
    assert v.n_history == 8 and v.status is Status.REGRESSION  # all 8 earlier points, the missing night adds nothing
    assert evaluate(s, TH, night=date(2025, 1, 8)).status is Status.MISSING


def test_improvement_lock_pct_is_configurable():
    vals = [100.0] * 12 + [106.0] * 8 + [100.0]
    default = evaluate(mk(vals), TH20)
    assert default.baseline_median == 100.0  # 6% gain < default 8%: no lock-in
    eager = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True, improvement_lock_pct=4.0)
    assert evaluate(mk(vals), eager).baseline_median == 106.0


def test_lock_window_is_configurable():
    vals = [100.0] * 16 + [120.0] * 4 + [105.0]
    assert evaluate(mk(vals), TH20).baseline_median == 100.0  # default 8: recent = 4x100 + 4x120, MAD too wide
    short = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True, lock_window=5)
    assert evaluate(mk(vals), short).baseline_median == 120.0  # last 5 = 100 + 4x120 -> median 120, MAD 0
    tiny = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True, lock_window=3)
    assert evaluate(mk(vals), tiny).baseline_median == 120.0  # below min_history: the last 5 points are used
    long = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True, lock_window=20)
    assert evaluate(mk(vals), long).baseline_median == 100.0


def test_several_points_per_night_tonight_is_the_last_and_earlier_runs_stay_in_baseline():
    start = date(2025, 1, 1)
    pts = [Point(start + timedelta(i), 10.0, 1, (f"r{i}", )) for i in range(8)]
    pts += [Point(start + timedelta(8), 10.0, 1, ("r8a", )), Point(start + timedelta(8), 8.0, 1, ("r8b", ))]
    v = evaluate(Series(KEY, tuple(pts)), TH)
    assert v.status is Status.REGRESSION and v.n_history == 9
