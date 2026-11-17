"""Tests for suppress.apply (synthetic data only)."""
from datetime import date, timedelta

import pytest

from benchmark_monitor import SeriesKey, Status, Thresholds, Verdict
from benchmark_monitor.suppress import ConfirmedFamily, SuppressConfig, SuppressionResult, apply, is_flip_flop

D0 = date(2026, 11, 1)
TH = Thresholds(5.0, 3.0, 5, 20, 15.0, 0.05, True)


def n(i):
    return D0 + timedelta(days=i)


def verdict(i, case="c1", bench="gemm", source="micro", status=Status.REGRESSION, drop=10.0):
    key = SeriesKey(source, "b580", "client", bench, "triton", "tflops", f'{{"case":"{case}"}}')
    return Verdict(key, n(i), status, drop, -4.0, 10.0, 0.1, 20, TH)


def history(flags, night, *, checked=100, override=None, source="micro", skip=()):
    """Per-night verdicts for nights 0..night: *flags* plus OK fillers so that `checked` series ran."""
    out = {}
    for i in range(night + 1):
        if i in skip:
            continue
        fl = [f for f in flags if f.night == n(i)]
        total = (override or {}).get(i, checked)
        out[n(i)] = fl + [verdict(i, f"ok{j}", "filler", source, Status.OK) for j in range(total - len(fl))]
    return out


def run(flags, night, cfg=None, **kw):
    return apply(history(flags, night, **kw), cfg or SuppressConfig(), n(night))


def run_nights(flags, nights, cfg=None, **kw):
    """`apply` for each night, seeing only the flags up to that night (as the nightly would)."""
    ov = kw.pop("override", None)
    return {i: run([f for f in flags if f.night <= n(i)], i, cfg, override=ov, **kw) for i in nights}


def test_default_knobs():
    cfg = SuppressConfig()
    assert (cfg.storm_threshold, cfg.confirm_nights, cfg.min_coverage, cfg.max_gap_days) == (10, 2, 0.7, 3)
    assert (cfg.flip_episodes, cfg.flip_days) == (3, 28)


def test_a_few_new_flags_are_shown_and_not_confirmed_on_night_one():
    res = run([verdict(10, "a"), verdict(10, "b")], 10)
    assert len(res.new) == 2 and not res.confirmed and res.item_count == 2 and not res.storms


def test_second_night_confirms_exactly_once():
    flags = [verdict(9), verdict(10), verdict(11)]
    assert [c.verdict.night for c in run(flags[:2], 10).confirmed] == [n(10)]
    third = run(flags, 11)
    assert not third.confirmed and third.item_count == 0 and len(third.sustained) == 1
    assert run(flags[:2], 10).confirmed[0].first_night == n(9)


def test_sustained_excludes_confirmed_items():
    res = run([verdict(9), verdict(10)], 10)
    assert len(res.confirmed) == 1 and not res.sustained


def test_confirm_nights_one_links_on_the_first_night():
    res = run([verdict(10)], 10, SuppressConfig(confirm_nights=1))
    assert len(res.confirmed) == 1 and res.item_count == 1  # no double count


def test_storm_folds_rows_into_a_family_digest_and_spares_other_families():
    fam = {"gemm-at": "gemm", "flex-attn-masks": "flex-attn", "softmax": "softmax"}
    flags = [verdict(10, f"c{i}", "gemm-at", drop=6.0 + i) for i in range(10)]
    flags += [verdict(10, f"f{i}", "flex-attn-masks") for i in range(3)] + [verdict(10, "s", "softmax")]
    res = run(flags, 10, SuppressConfig(family_of=lambda b: fam[b]))
    assert [(d.family, d.count) for d in res.storms] == [("gemm", 10)]
    assert res.storms[0].worst_drop == 15.0 and res.storms[0].median_drop == 10.5
    assert sorted(v.key.benchmark for v in res.new) == ["flex-attn-masks"] * 3 + ["softmax"]  # not hidden
    assert res.item_count == 5 and res.storm_nights == {n(10)}


def test_threshold_is_per_family_not_global():
    fam = {"a": "fa", "b": "fb"}
    flags = [verdict(10, f"a{i}", "a") for i in range(6)] + [verdict(10, f"b{i}", "b") for i in range(6)]
    res = run(flags, 10, SuppressConfig(family_of=fam.get))
    assert not res.storms and len(res.new) == 12  # 12 flags in total, but only 6 per family


def test_storm_counts_live_flags_only_on_every_night():
    """10 flags from a partial run are held back, so that night is not a storm night."""
    flags = [verdict(i, f"c{j}") for i in (9, 10) for j in range(10)]
    res = run_nights(flags, (9, 10), override={9: 20}, checked=100)
    assert res[9].suspect and res[9].storms == () and res[9].new == ()
    assert n(9) not in res[10].storm_nights and n(9) not in res[9].storm_nights


def test_storm_can_be_disabled():
    res = run([verdict(10, f"c{i}") for i in range(12)], 10, SuppressConfig(storm_threshold=0))
    assert not res.storms and len(res.new) == 12


def test_episode_that_started_in_a_storm_is_never_confirmed():
    flags = [verdict(i, f"c{i2}") for i in (9, 10) for i2 in range(10)]
    assert not run(flags, 10).confirmed
    assert len(run(flags, 10, SuppressConfig(storm_threshold=0)).confirmed) == 10


def test_partial_run_holds_back_flags():
    res = run([verdict(10, "a"), verdict(10, "b")], 10, override={10: 40})
    assert not res.new and res.held == {"micro": 2} and res.item_count == 0
    assert "only 40 series" in res.suspect["micro"] and any("held back" in w for w in res.warnings)


def test_partial_run_guard_needs_enough_history_and_can_be_disabled():
    few = {n(i): history([], 10, override={10: 1})[n(i)] for i in (8, 9, 10)}
    assert apply(few, SuppressConfig(), n(10)).suspect == {}
    assert run([], 10, override={10: 1}, cfg=SuppressConfig(min_coverage=0)).suspect == {}
    off = SuppressConfig(min_coverage=0, max_gap_days=0)
    assert run([], 10, override={10: 1}, cfg=off).suspect == {}


def test_data_gap_marks_the_source_suspect():
    h = history([verdict(10)], 10, skip=range(5, 10))
    assert "6 days back" in apply(h, SuppressConfig(), n(10)).suspect["micro"]
    assert apply(h, SuppressConfig(max_gap_days=0), n(10)).suspect == {}


def test_no_data_tonight_is_not_suspect_but_warns():
    res = apply(history([], 9), SuppressConfig(), n(10))
    assert res.suspect == {} and "latest data night" in res.warnings[0]


def test_a_night_without_data_does_not_make_a_standing_flag_new_again():
    h = history([verdict(8), verdict(10)], 10, skip=(9, ))
    res = apply(h, SuppressConfig(), n(10))
    assert not res.new and len(res.confirmed) == 1 and not res.sustained  # confirmed, not also sustained


def test_flip_flop_cases_are_muted():
    flags = [verdict(i, "flaky") for i in (2, 5, 8, 11)] + [verdict(i, "steady") for i in (8, 9, 10, 11)]
    res = run(flags, 11)
    assert [m.key.params for m in res.muted] == ['{"case":"flaky"}'] and not res.new
    assert all(v.key not in {m.key for m in res.muted} for v in res.sustained)  # muted is not also sustained
    assert res.item_count == 0


def test_flip_flop_needs_enough_episodes_and_mostly_short_ones():

    def muted(nights, **cfg):
        return run([verdict(i) for i in nights], 11, SuppressConfig(**cfg)).muted

    assert not muted((5, 11))
    assert not muted((1, 2, 3, 6, 7, 8, 10, 11))  # long episodes
    assert not muted((2, 5, 8, 11), flip_episodes=0)
    assert len(muted((2, 5, 8, 11))) == 1
    assert not muted((2, 5, 8, 11), flip_days=4)


def test_regression_after_earlier_blips_is_not_muted_and_is_confirmed():
    """Blips on nights 2 and 5, then a real regression from night 8, is not muted."""
    flags = [verdict(i) for i in (2, 5, 8, 9, 10, 11, 12, 13)]
    res = run_nights(flags, range(8, 14))
    assert [r.item_count for r in res.values()] == [1, 1, 0, 0, 0, 0]  # watch on 8, confirmed on 9
    assert [len(r.confirmed) for r in res.values()] == [0, 1, 0, 0, 0, 0]
    assert all(r.muted == () for r in res.values())


def test_mute_needs_completed_episodes_and_lifts_on_the_second_night():
    res = run_nights([verdict(i) for i in (1, 3, 5, 8, 9, 10)], range(8, 11))
    assert res[8].muted != () and res[8].item_count == 0
    assert res[9].muted == () and len(res[9].confirmed) == 1


def test_a_muted_sustained_case_is_listed_as_muted():
    assert not run([verdict(i) for i in (1, 3, 5, 7, 8, 9)], 9).muted  # streak 3 > longest earlier: lifted
    res = run([verdict(i) for i in (0, 1, 2, 4, 6, 8, 9)], 9, SuppressConfig(flip_short_share=0.3))
    assert [v.night for v in res.muted] == [n(9)] and not res.new  # streak 2 <= longest earlier 3
    assert res.sustained == ()  # a muted case is not also sustained


def test_confirmation_survives_a_partial_run_on_the_confirming_night():
    """Night 9 (the confirming night) was a partial run; night 10 must confirm."""
    res = run_nights([verdict(i) for i in (8, 9, 10, 11)], range(9, 12), override={9: 10})
    assert res[9].confirmed == () and "micro" in res[9].suspect
    assert len(res[10].confirmed) == 1  # first non-suspect night with streak >= 2
    assert res[11].confirmed == ()  # and only once


def test_confirmation_is_not_repeated_when_the_confirming_night_alerted():
    res = run_nights([verdict(i) for i in (8, 9, 10, 11)], range(9, 12))
    assert [len(r.confirmed) for r in res.values()] == [1, 0, 0]


def test_storm_started_episodes_stay_unconfirmed_on_later_nights():
    flags = [verdict(i, f"s{j}") for i in (9, 10, 11) for j in range(10)]
    assert all(r.confirmed == () for r in run_nights(flags, range(10, 12)).values())


def gemm_flags(nights, cases=12):
    return [verdict(i, f"g{j:02d}", "gemm") for i in nights for j in range(cases)]


def test_storm_on_the_first_history_night_still_counts():
    flags = gemm_flags((0, 1, 2), cases=10)
    res = run_nights(flags, (0, 1, 2))
    assert [r.storms[0].count if r.storms else 0 for r in res.values()] == [10, 0, 0]
    assert all(r.confirmed == () for r in res.values())  # not confirmed one by one
    assert n(0) in res[1].storm_nights and n(0) in res[2].storm_nights
    assert [len(r.confirmed_families) for r in res.values()] == [0, 1, 0]


def test_consecutive_nights_do_not_repeat_a_confirmation_as_old_episodes_leave_the_window():
    """Episodes on days 0-1, 3-4 and 6 leave the 28-day window while the day 26 episode runs."""
    flags = [verdict(i) for i in (0, 1, 3, 4, 6, 26, 27, 28)]
    res = run_nights(flags, range(26, 29))
    assert [len(r.confirmed) for r in res.values()] == [0, 1, 0]
    assert res[27].confirmed[0].first_night == n(26)


def test_broad_regression_is_a_digest_then_one_family_confirmation():
    res = run_nights(gemm_flags((9, 10, 11, 12)), range(9, 13))
    assert [(len(r.storms), len(r.confirmed_families)) for r in res.values()] == [(1, 0), (0, 1), (0, 0), (0, 0)]
    assert all(r.confirmed == () for r in res.values())
    cf = res[10].confirmed_families[0]
    assert (cf.source, cf.family, cf.night, cf.first_night) == ("micro", "gemm", n(10), n(9))
    assert cf.keys == tuple(sorted(cf.keys, key=lambda k: k.params)) and len(cf.keys) == 12
    assert isinstance(cf, ConfirmedFamily) and res[10].item_count == 1


def test_family_confirmation_lists_only_cases_still_regressing():
    flags = gemm_flags((9,)) + gemm_flags((10,), cases=3)
    cf = run(flags, 10).confirmed_families[0]
    assert [k.params for k in cf.keys] == [f'{{"case":"g{j:02d}"}}' for j in range(3)]


def test_storm_that_recovers_by_the_next_night_confirms_no_family():
    res = run_nights(gemm_flags((9,)), (9, 10, 11))
    assert all(r.confirmed_families == () and r.confirmed == () for r in res.values())


def test_family_confirmation_survives_a_missed_night():
    flags = gemm_flags((9, 11))
    res = run(flags, 11, skip=(10, ))
    assert [len(res.confirmed_families), res.confirmed_families[0].first_night] == [1, n(9)]
    assert run(gemm_flags((9, 11, 12)), 12, skip=(10, )).confirmed_families == ()  # only once


def test_family_confirmation_survives_a_partial_run_on_the_confirming_night():
    res = run_nights(gemm_flags((9, 10, 11, 12)), range(10, 13), override={10: 10})
    assert res[10].confirmed_families == () and "micro" in res[10].suspect
    assert len(res[11].confirmed_families) == 1 and res[12].confirmed_families == ()


def test_family_confirmation_respects_confirm_nights():
    cfg = SuppressConfig(confirm_nights=3)
    res = run_nights(gemm_flags((9, 10, 11, 12)), range(10, 13), cfg)
    assert [len(r.confirmed_families) for r in res.values()] == [0, 1, 0]


def test_confirm_nights_of_three_confirms_a_case_on_the_third_night():
    res = run_nights([verdict(i) for i in (8, 9, 10, 11)], range(9, 12), SuppressConfig(confirm_nights=3))
    assert [len(r.confirmed) for r in res.values()] == [0, 1, 0]


def test_coverage_history_limits_the_baseline_and_zero_turns_the_check_off():
    ov = {i: 100 for i in range(10)} | {3: 20, 4: 20, 5: 20, 6: 20, 7: 20, 8: 20, 9: 20, 10: 40}
    assert run([], 10, override=ov).suspect == {}  # median of the last 10 nights is 20
    assert "micro" in run([], 10, SuppressConfig(coverage_history=3), override={**ov, 7: 100, 8: 100, 9: 100}).suspect
    assert run([], 10, SuppressConfig(coverage_history=0), override={**ov, 10: 1}).suspect == {}


def test_invalid_knobs_are_rejected():
    for bad in ({"storm_threshold": -1}, {"coverage_history": -1}, {"min_coverage": 1.5}, {"flip_short_share": -0.1}):
        with pytest.raises(ValueError):
            SuppressConfig(**bad)


def test_empty_history_and_pure_flip_flop_rule():
    assert apply({}, SuppressConfig(), n(3)) == SuppressionResult()
    assert not is_flip_flop([3, 3, 3], 1, SuppressConfig())
    assert is_flip_flop([1, 1, 3], 1, SuppressConfig())
    assert not is_flip_flop([1, 1, 1], 2, SuppressConfig())  # longer than any earlier episode


def missing_history(spec, night):
    """One series ("c1") whose verdict status per night is spec[i]; fillers keep the source alive."""
    out = {}
    for i in range(night + 1):
        key_v = verdict(i, "c1", status=spec.get(i, Status.OK))
        out[n(i)] = [key_v] + [verdict(i, f"ok{j}", "filler", status=Status.OK) for j in range(100)]
    return out


def test_series_own_missing_night_does_not_break_its_episode():
    R, M = Status.REGRESSION, Status.MISSING
    spec = {3: R, 4: M, 5: R, 6: R}
    res = {
        i: apply({d: v
                  for d, v in missing_history(spec, 6).items()
                  if d <= n(i)}, SuppressConfig(), n(i))
        for i in (3, 5, 6)
    }
    assert len(res[3].new) == 1 and not res[3].confirmed
    assert not res[5].new and not res[5].sustained  # not new again; listed as confirmed only
    assert len(res[5].confirmed) == 1 and res[5].confirmed[0].first_night == n(3)
    assert not res[6].confirmed


def test_confirm_nights_below_one_is_clamped():
    res = run([verdict(10)], 10, SuppressConfig(confirm_nights=0))
    assert len(res.confirmed) == 1
