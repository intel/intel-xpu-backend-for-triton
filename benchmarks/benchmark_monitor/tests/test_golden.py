"""Run the gate on every golden case (synthetic corpus, see golden/README.md)."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from benchmark_monitor import Observation, Series, SeriesKey, Status, Thresholds
from benchmark_monitor.aggregate import aggregate
from benchmark_monitor.gate import evaluate

GOLDEN = Path(__file__).parent / "golden" / "cases.jsonl"
CASES = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line]
FIELDS = ("drop_pct", "z", "baseline_median", "mad")


def run_gate(case: dict):
    key = SeriesKey("micro", "b580", "ci", case["benchmark"], "triton", case["metric"], "{}")
    as_of = date.fromisoformat(case["as_of"])
    obs = [
        Observation(key, o["run"],
                    datetime.fromisoformat(o["night"]).replace(tzinfo=timezone.utc), o["value"])
        for o in case["observations"]
        if o["value"] > 0  # the source layer drops non-positive measurements before they reach the gate
    ]  # observations after as_of stay in: evaluate(night=as_of) must ignore them itself
    series = aggregate(obs, case["unit"]) if obs else Series(key, ())
    return evaluate(series, Thresholds(**case["thresholds"]), night=as_of)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_gate_matches_expected(case):
    verdict = run_gate(case)
    exp = case["expected"]
    assert verdict.status is Status(exp["status"])
    assert verdict.n_history == exp["n_history"]
    for name in FIELDS:
        got = getattr(verdict, name)
        if exp[name] is None:
            assert got is None, name
        else:
            assert got == pytest.approx(exp[name], abs=1e-6), name


def test_corpus_shape():
    assert len(CASES) == 47
    assert sum(c["fp_sensitive"] for c in CASES) == 7
    assert len({c["id"] for c in CASES}) == len(CASES)


@pytest.mark.parametrize("case", [c for c in CASES if c["references_disagree"]], ids=lambda c: c["id"])
def test_documented_differences_are_real(case):
    """Each documented difference names a #6491 reference whose status really differs from expected."""
    for ref in case["disagreeing_references"]:
        assert case["references"][ref]["status"] != case["expected"]["status"]
    assert case["disagreement_cause"]
