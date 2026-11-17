"""Tests for pipeline.run with a fake source and a fake gate."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from benchmark_monitor import Observation, SeriesKey, Status, Verdict, load_catalog
from benchmark_monitor.pipeline import run
from benchmark_monitor.suppress import SuppressConfig

AS_OF = date(2026, 11, 10)


class FakeSource:

    def __init__(self, benchmarks, flagged_from=7):
        self.benchmarks, self.flagged_from, self.calls = benchmarks, flagged_from, []

    def observations(self, scope, start, end):
        self.calls.append((scope.name, start, end))
        for b in self.benchmarks:
            key = SeriesKey(scope.source, scope.platform, "client", b, "triton", "tflops", "{}")
            for i in range((end - start).days + 1):
                day = start + timedelta(days=i)
                low = day.day >= self.flagged_from
                yield Observation(key, f"r{i}", datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc),
                                  5.0 if low else 10.0)


def fake_gate(series, thresholds, night):
    """Regression whenever the night's value is below 8; stands in for gate.evaluate."""
    val = {p.night: p.value for p in series.points}.get(night)
    status = Status.MISSING if val is None else Status.REGRESSION if val < 8 else Status.OK
    return Verdict(series.key, night, status, 50.0 if status is Status.REGRESSION else 0.0, -5.0, 10.0, 0.1, 5,
                   thresholds)


def test_run_builds_history_applies_suppression_and_uses_catalog_families():
    cat = load_catalog()
    src = FakeSource(["flash-attn-a", "batched-flash-attn-b"])
    rep = run(cat, src, ["nightly-b580-micro"], AS_OF, 6, gate=fake_gate)
    assert src.calls == [("nightly-b580-micro", date(2026, 9, 6), AS_OF)]  # first night - 3 x max window
    assert rep.as_of == AS_OF and len(rep.verdicts) == 2
    assert all(v.status is Status.REGRESSION for v in rep.verdicts)
    assert rep.suppression.new == () and len(rep.suppression.sustained) == 2  # flagged since night 7
    assert len(rep.suppression.confirmed) == 0  # confirmed back on the 8th


def test_storm_uses_catalog_family_and_custom_cfg():
    cat = load_catalog()
    src = FakeSource([f"flash-attn-{i}" for i in range(3)], flagged_from=10)
    rep = run(cat, src, ["nightly-b580-micro"], AS_OF, 6, SuppressConfig(storm_threshold=3), gate=fake_gate)
    assert [(d.family, d.count) for d in rep.suppression.storms] == [("flash-attn", 3)]


def test_no_observations_warns():
    rep = run(load_catalog(), FakeSource([]), ["nightly-b580-micro"], AS_OF, 3, gate=fake_gate)
    assert not rep.verdicts and any("no observations" in w for w in rep.warnings)


class StepSource:
    """Flat 10.0 with +-0.1 jitter, dropping to 7.0 from ``step`` on."""

    def __init__(self, step):
        self.step = step

    def observations(self, scope, start, end):
        key = SeriesKey(scope.source, scope.platform, "client", "softmax", "triton", "tflops", "{}")
        for i in range((end - start).days + 1):
            day = start + timedelta(days=i)
            value = 7.0 if day >= self.step else 10.0 + 0.1 * (-1)**i
            yield Observation(key, f"r{i}", datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc), value)


def test_real_gate_new_then_confirmed():
    cat, src = load_catalog(), StepSource(AS_OF)
    first = run(cat, src, ["nightly-b580-micro"], AS_OF, 15)
    assert [v.status for v in first.verdicts] == [Status.REGRESSION]
    assert len(first.suppression.new) == 1 and not first.suppression.confirmed
    second = run(cat, src, ["nightly-b580-micro"], AS_OF + timedelta(days=1), 15)
    assert not second.suppression.new and len(second.suppression.confirmed) == 1


def test_nights_must_be_positive_and_short_lookback_warns():
    with pytest.raises(ValueError, match="at least 1"):
        run(load_catalog(), FakeSource([]), ["nightly-b580-micro"], AS_OF, 0, gate=fake_gate)
    rep = run(load_catalog(), StepSource(AS_OF), ["nightly-b580-micro"], AS_OF, 5)
    assert any("flip-flop look-back" in w for w in rep.warnings)


def test_baseline_history_is_fetched_before_the_evaluated_nights():
    rep = run(load_catalog(), StepSource(AS_OF), ["nightly-b580-micro"], AS_OF, 1)
    assert [v.status for v in rep.verdicts] == [Status.REGRESSION] and rep.verdicts[0].n_history == 20


def test_undeclared_metric_is_skipped_with_one_warning():

    class Odd(StepSource):

        def observations(self, scope, start, end):
            for o in super().observations(scope, start, end):
                yield o
                yield Observation(replace(o.key, metric="furlongs"), o.run_id, o.observed_at, o.value)

    rep = run(load_catalog(), Odd(AS_OF), ["nightly-b580-micro"], AS_OF, 3)
    assert len(rep.verdicts) == 1 and sum("furlongs" in w for w in rep.warnings) == 1


def test_overlapping_scopes_do_not_double_count():
    cat = load_catalog()
    seen = []

    def counting_gate(series, thresholds, night):
        seen.append(sum(p.n_raw for p in series.points if p.night == night))
        return fake_gate(series, thresholds, night)

    src = FakeSource(["gemm"])
    run(cat, src, ["nightly-b580-micro", "nightly-b580-micro"], AS_OF, 1, gate=counting_gate)
    assert seen == [1]  # the same observation, selected by both scopes, counts once


def test_distinct_runs_of_one_series_are_kept_across_scopes():
    seen = []

    def counting_gate(series, thresholds, night):
        seen.append(sum(p.n_raw for p in series.points if p.night == night))
        return fake_gate(series, thresholds, night)

    class Two:

        def observations(self, scope, _start, _end):
            key = SeriesKey("micro", "b580", "client", "gemm", "triton", "tflops", "{}")
            moment = datetime(2026, 11, 10, 3, tzinfo=timezone.utc)
            # scope a returns runs A and B; scope b only A
            yield from [
                Observation(key, r, moment, v)
                for r, v in (("A", 10.0), ("B", 4.0))
                if scope.name == "nightly-b580-micro" or r == "A"
            ]

    cat = load_catalog()
    other = replace(cat.scopes["nightly-b580-micro"], name="other")
    cat = replace(cat, scopes={**cat.scopes, "other": other})
    run(cat, Two(), ["nightly-b580-micro", "other"], AS_OF, 1, gate=counting_gate)
    assert seen == [2]  # A (shared) + B, not A + A + B


@pytest.mark.parametrize("nights,warns", [(1, True), (2, False)])
def test_few_nights_warn_that_nothing_can_be_confirmed(nights, warns):
    rep = run(load_catalog(), FakeSource(["gemm"]), ["nightly-b580-micro"], AS_OF, nights, gate=fake_gate)
    assert any("nothing can be confirmed" in w for w in rep.warnings) is warns
