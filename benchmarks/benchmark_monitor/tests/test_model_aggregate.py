"""Tests for model and aggregate (synthetic data only)."""
import time
from datetime import date, datetime, timedelta, timezone

import pytest

from benchmark_monitor import Observation, Series, SeriesKey, Status, aggregate, canonical_params

KEY = SeriesKey("micro", "b580", "client", "softmax", "triton", "tflops", canonical_params({"N": 1, "M": 2}))


def obs(run, when, value, key=KEY):
    return Observation(key, run, when, value, {"triton_commit": "abc"})


def test_canonical_params_sorted_no_spaces():
    assert KEY.params == '{"M":2,"N":1}'


def test_status_values():
    assert Status("noisy") is Status.NOISY
    assert {s.value for s in Status} == {"regression", "improvement", "ok", "insufficient_history", "noisy", "missing"}


def test_model_is_frozen():
    with pytest.raises(AttributeError):
        KEY.tag = "ci"  # type: ignore[misc]


def test_night_mean_merges_duplicates_and_sorts():
    d1 = datetime(2026, 11, 2, 1, tzinfo=timezone.utc)
    s = aggregate(
        [obs("r2", d1 + timedelta(hours=2), 30.0),
         obs("r1", d1, 10.0),
         obs("r0", d1 - timedelta(days=1), 5.0)])
    assert isinstance(s, Series) and s.key == KEY
    assert [p.night for p in s.points] == [date(2026, 11, 1), date(2026, 11, 2)]
    p = s.points[1]
    assert (p.value, p.n_raw, p.run_ids) == (20.0, 2, ("r1", "r2"))


def test_night_mean_is_mean_of_run_means():
    t = datetime(2026, 11, 2, 1, tzinfo=timezone.utc)
    s = aggregate([obs("A", t, 10.0), obs("A", t, 20.0), obs("B", t + timedelta(hours=2), 40.0)])
    p = s.points[0]
    assert (p.value, p.n_raw, p.run_ids) == (27.5, 3, ("A", "B"))  # not the row mean 23.33


def test_night_is_utc_and_naive_is_utc():
    late_est = datetime(2026, 11, 1, 22, tzinfo=timezone(timedelta(hours=-5)))  # 03:00 UTC next day
    s = aggregate([obs("a", late_est, 1.0), obs("b", datetime(2026, 11, 2, 4), 3.0)])
    assert len(s.points) == 1 and s.points[0].night == date(2026, 11, 2) and s.points[0].value == 2.0


def test_run_unit_one_point_per_run():
    t = datetime(2026, 11, 2, 1)
    s = aggregate([obs("r1", t, 10.0), obs("r1", t, 20.0), obs("r2", t + timedelta(hours=3), 7.0)], unit="run")
    assert [(p.run_ids, p.value, p.n_raw) for p in s.points] == [(("r1", ), 15.0, 2), (("r2", ), 7.0, 1)]


def test_rejects_empty_mixed_keys_and_bad_unit():
    t = datetime(2026, 11, 2)
    with pytest.raises(ValueError):
        aggregate([])
    other = SeriesKey("micro", "b580", "client", "gemm", "triton", "tflops", "{}")
    with pytest.raises(ValueError):
        aggregate([obs("a", t, 1.0), obs("b", t, 1.0, other)])
    with pytest.raises(ValueError):
        aggregate([obs("a", t, 1.0)], unit="week")  # type: ignore[arg-type]


def test_naive_datetime_is_utc_not_host_local(monkeypatch):
    # 01:00 naive is 2026-11-02 in UTC but 2026-11-01 (19:30 UTC) if read as Asia/Kolkata local time.
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    time.tzset()
    try:
        s = aggregate([obs("a", datetime(2026, 11, 2, 1), 1.0)])
    finally:
        monkeypatch.undo()
        time.tzset()
    assert s.points[0].night == date(2026, 11, 2)


def test_run_unit_orders_out_of_order_input_by_time():
    t = datetime(2026, 11, 2, 1)
    s = aggregate(
        [obs("late", t + timedelta(days=1), 3.0),
         obs("early", t, 1.0),
         obs("mid", t + timedelta(hours=5), 2.0)], unit="run")
    assert [p.run_ids for p in s.points] == [("early", ), ("mid", ), ("late", )]


def test_observation_attrs_are_read_only_and_drop_empty_strings():
    source = {"triton_commit": "abc", "driver": ""}
    o = Observation(KEY, "r", datetime(2026, 11, 2), 1.0, source)
    assert dict(o.attrs) == {"triton_commit": "abc"}
    with pytest.raises(TypeError):
        o.attrs["x"] = "y"  # type: ignore[index]
    source["later"] = "z"
    assert "later" not in o.attrs


def test_observation_is_hashable():
    o = Observation(KEY, "r", datetime(2026, 11, 2), 1.0, {"a": "b"})
    assert hash(o) == hash(Observation(KEY, "r", datetime(2026, 11, 2), 1.0, {"a": "c"}))
