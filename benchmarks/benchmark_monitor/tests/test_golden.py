"""Run the gate on every golden case (synthetic corpus, see golden/README.md)."""

from __future__ import annotations

import dataclasses
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


def _observed_at(text: str) -> datetime:
    """Timestamp of a corpus observation: a date or ISO timestamp; an offset is kept, a naive value is UTC."""
    moment = datetime.fromisoformat(text)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def run_gate(case: dict):
    key = SeriesKey("micro", "b580", "ci", case["benchmark"], "triton", case["metric"], "{}")
    as_of = date.fromisoformat(case["as_of"])
    obs = [
        Observation(key, o["run"], _observed_at(o["night"]), o["value"])
        for o in case["observations"]
        if o["value"] > 0  # the source drops <= 0 for unsigned metrics (all corpus metrics are unsigned)
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


CASE_KEYS = {
    "id", "description", "tags", "benchmark", "metric", "unit", "as_of", "observations", "thresholds", "expected",
    "fp_sensitive", "references", "references_disagree"
}
OPTIONAL_KEYS = {"notes", "disagreeing_references", "disagreement_cause"}
VERDICT_KEYS = {"status", "drop_pct", "z", "baseline_median", "mad", "n_history"}


def test_corpus_shape():
    assert len({c["id"] for c in CASES}) == len(CASES)
    assert any(c["fp_sensitive"] for c in CASES)
    readme = (GOLDEN.parent / "README.md").read_text(encoding="utf-8")
    assert f"holds {len(CASES)} synthetic cases" in readme  # the README count follows the file


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_case_schema(case):
    assert CASE_KEYS <= set(case) <= CASE_KEYS | OPTIONAL_KEYS
    assert set(case["thresholds"]) == {f.name for f in dataclasses.fields(Thresholds)}
    assert set(case["expected"]) == VERDICT_KEYS
    assert case["unit"] in ("night_mean", "run")
    assert all(set(o) == {"run", "night", "value"} for o in case["observations"])
    assert case["fp_sensitive"] == ("fp" in case["tags"])
    assert case["references_disagree"] == ("disagreeing_references" in case)
    assert ("disagreeing_references" in case) == ("disagreement_cause" in case)
    assert set(case["references"]) == {"pr6491_per_run", "pr6491_night_mean"}
    for ref in case["references"].values():
        assert VERDICT_KEYS <= set(ref)


def _differs(ref: dict, expected: dict) -> bool:
    for name in VERDICT_KEYS:
        a, b = ref[name], expected[name]
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > 1e-6:
                return True
        elif a != b:
            return True
    return False


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_documented_differences_are_real(case):
    """The references that differ from expected are exactly the listed ones, and a difference has a cause."""
    differing = sorted(name for name, ref in case["references"].items() if _differs(ref, case["expected"]))
    assert differing == sorted(case.get("disagreeing_references", []))
    assert case["references_disagree"] == bool(differing)
    if differing:
        assert case["disagreement_cause"]
