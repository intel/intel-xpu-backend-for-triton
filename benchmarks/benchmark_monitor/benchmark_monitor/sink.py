"""Report sinks: Markdown and JSON, writing to a path or an open text stream."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from datetime import date
from typing import IO, Any, Iterable, Iterator, Protocol, TypedDict

from benchmark_monitor.model import SeriesKey, Status, Verdict
from benchmark_monitor.report import Report
from benchmark_monitor.suppress import Confirmed, SuppressionResult


class Sink(Protocol):
    """Consumes a finished report."""

    def emit(self, report: Report) -> None:
        ...


@contextmanager
def _open(target: str | Path | IO[str]) -> Iterator[IO[str]]:
    if isinstance(target, (str, Path)):
        with open(target, "w", encoding="utf-8") as handle:
            yield handle
    else:
        yield target


def _fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


MAX_FAMILY_CASES = 3  # top cases named per confirmed family
MAX_LISTED = 10  # rows listed per suppression section; the rest is summarised as "... and N more"


def _case(key: SeriesKey) -> tuple[str, ...]:
    """A case is a series without its metric: all metrics of one benchmark/params/platform share a row."""
    return (key.source, key.platform, key.tag, key.benchmark, key.compiler, key.params)


def _label(key: SeriesKey) -> str:
    return f"{key.benchmark} / {key.platform} / {key.compiler} `{key.params}`"


def _labels(keys: Iterable[SeriesKey]) -> dict[tuple[str, ...], str]:
    """Label per case. Source and tag are added only where two cases would otherwise print the same label."""
    cases = {_case(k): k for k in keys}
    base = Counter(_label(k) for k in cases.values())
    return {c: _label(k) + (f" [{k.source}, {k.tag}]" if base[_label(k)] > 1 else "") for c, k in cases.items()}


def _by_case(verdicts: Iterable[Verdict]) -> list[tuple[str, list[Verdict]]]:
    """(label, verdicts) per case, worst drop first, then benchmark/params; metrics ordered the same way."""
    groups: dict[tuple[str, ...], list[Verdict]] = defaultdict(list)
    for v in verdicts:
        groups[_case(v.key)].append(v)
    for vs in groups.values():
        vs.sort(key=lambda v: (-(v.drop_pct or 0.0), v.key.metric))
    order = sorted(groups.items(), key=lambda kv: (-(kv[1][0].drop_pct or 0.0), kv[0][3], kv[0][5], kv[0]))
    labels = _labels(v.key for vs in groups.values() for v in vs)
    return [(labels[case], vs) for case, vs in order]


def n_cases(verdicts: Iterable[Verdict]) -> int:
    return len({_case(v.key) for v in verdicts})


def _short_list(title: str, items: list[str], series: int | None = None) -> list[str]:
    """A capped bullet list; the header counts items (cases), plus the series count if it differs."""
    if not items:
        return []
    extra = f", {series} series" if series is not None and series != len(items) else ""
    lines = ["", f"## {title} ({len(items)}{extra})", ""] + [f"- {i}" for i in items[:MAX_LISTED]]
    if len(items) > MAX_LISTED:
        lines.append(f"- ... and {len(items) - MAX_LISTED} more")
    return lines


def _change(v: Verdict) -> str:
    """Signed change of the metric itself: ``drop_pct`` is positive-is-worse, so a rise of a lower-is-better
    metric prints ``+20.0%`` and a fall of a higher-is-better one prints ``-9.1%``."""
    if v.drop_pct is None:
        return "-"
    return format(-v.drop_pct if v.thresholds.higher_is_better else v.drop_pct, "+.1f") + "%"


def _flagged(verdicts: tuple[Verdict, ...]) -> tuple[list[str], int]:
    rows = [f"{label}: " + ", ".join(f"{v.key.metric} {_change(v)}" for v in vs) for label, vs in _by_case(verdicts)]
    return rows, len(verdicts)


def _confirmed(sup: SuppressionResult) -> tuple[list[str], int]:
    groups: dict[tuple[str, ...], list[Confirmed]] = defaultdict(list)
    for c in sup.confirmed:
        groups[_case(c.key)].append(c)
    rows = []
    labels = _labels(c.key for c in sup.confirmed)
    for case, cs in sorted(groups.items(), key=lambda kv: (kv[1][0].key.benchmark, kv[1][0].key.params, kv[0])):
        cs.sort(key=lambda c: c.key.metric)
        rows.append(f"{labels[case]}: " +
                    ", ".join(f"{c.key.metric} {c.streak} nights since {c.first_night}" for c in cs))
    return rows, len(sup.confirmed)


class FamilyRow(TypedDict):
    """One confirmed family in the reports: the grouped view (cases) plus the series ``keys`` it came from."""

    family: str
    source: str
    night: date
    first_night: date
    n_cases: int
    top_cases: list[str]
    keys: list[dict[str, Any]]


def _families(sup: SuppressionResult) -> list[FamilyRow]:
    """Confirmed storm families with a case count, top cases and the series keys."""
    out: list[FamilyRow] = []
    for cf in sup.confirmed_families:
        labels = _labels(cf.keys)
        groups: dict[tuple[str, ...], SeriesKey] = {}
        for key in cf.keys:
            groups.setdefault(_case(key), key)
        ordered = sorted(groups.items(), key=lambda kv: (kv[1].benchmark, kv[1].params, kv[0]))
        out.append({
            "family": cf.family,
            "source": cf.source,
            "night": cf.night,
            "first_night": cf.first_night,
            "n_cases": len(ordered),
            "top_cases": [labels[c] for c, _ in ordered[:MAX_FAMILY_CASES]],
            "keys": [asdict(k) for k in cf.keys],
        })
    return sorted(out, key=lambda d: (d["source"], d["family"], str(d["night"])))


def _confirmed_families(sup: SuppressionResult) -> list[str]:
    rows = []
    for d in _families(sup):
        tops, total = d["top_cases"], d["n_cases"]
        more = f", ... and {total - len(tops)} more" if total > len(tops) else ""
        family, source, night, first = d["family"], d["source"], d["night"], d["first_night"]
        rows.append(f"{family} ({source}): storm night {night}, first {first}, {total} cases; top: " + ", ".join(tops) +
                    more)
    return rows


def _suppression_lines(sup: SuppressionResult) -> list[str]:
    lines = _short_list("New", *_flagged(sup.new))
    lines += _short_list("Confirmed", *_confirmed(sup))
    lines += _short_list("Confirmed families", _confirmed_families(sup))
    lines += _short_list("Sustained", *_flagged(sup.sustained))
    lines += _short_list("Muted (flip-flop)", *_flagged(sup.muted))
    lines += _short_list(
        "Storms",
        [f"{d.family}: {d.count} new, median {d.median_drop:.1f}%, worst {d.worst_drop:.1f}%" for d in sup.storms])
    lines += _short_list("Suspect sources", [f"{src}: {why}" for src, why in sorted(sup.suspect.items())])
    return lines


def _json_default(obj: object) -> object:
    if isinstance(obj, (set, frozenset)):
        return sorted(str(o) for o in obj)
    return obj.value if hasattr(obj, "value") else str(obj)


def _cell(text: str) -> str:
    """Make *text* safe inside a Markdown table cell: no pipes, backticks or line breaks."""
    return " ".join(text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'").split())


class MarkdownSink:
    """Markdown report: the suppression sections if the report has a suppression view, else a regressions table;
    then counts per status and warnings."""

    def __init__(self, target: str | Path | IO[str]):
        self.target = target

    @staticmethod
    def _table(regs: list[Verdict]) -> list[str]:
        lines = [
            "| source | platform | tag | benchmark | compiler | metric | params | drop % | z | baseline | n |",
            "|---|---|---|---|---|---|---|---|---|---|---|"
        ]
        for v in sorted(regs, key=lambda v: -(v.drop_pct or 0.0)):
            k = v.key
            cells = tuple(_cell(c)
                          for c in (k.source, k.platform, k.tag, k.benchmark, k.compiler, k.metric, k.params)) + (_fmt(
                              v.drop_pct, ".1f"), _fmt(v.z, ".2f"), _fmt(v.baseline_median, ".4g"), str(v.n_history))
            lines.append("| " + " | ".join(cells) + " |")
        return lines

    def render(self, report: Report) -> str:
        regs = [v for v in report.verdicts if v.status is Status.REGRESSION]
        lines = [f"# Benchmark monitor report for {report.as_of.isoformat()}"]
        if report.suppression is not None:
            lines += [
                "", f"{n_cases(regs)} regressing cases tonight ({len(regs)} series); "
                f"{n_cases(report.suppression.new)} new."
            ]
            lines += _suppression_lines(report.suppression)
        else:
            lines += ["", f"## Regressions ({len(regs)})", ""]
            lines += self._table(regs) if regs else ["None."]
        counts = Counter(v.status for v in report.verdicts)
        lines += ["", "## Counts", ""] + [f"- {s.value}: {counts.get(s, 0)}" for s in Status]
        if report.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
        return "\n".join(lines) + "\n"

    def emit(self, report: Report) -> None:
        with _open(self.target) as out:
            out.write(self.render(report))


class JsonSink:
    """Full report as JSON (dates ISO-formatted, enums by value). Verdicts and suppression lists stay per series.
    ``confirmed_families`` rows add ``n_cases`` and ``top_cases`` (grouped by case) and keep every series ``keys``."""

    def __init__(self, target: str | Path | IO[str]):
        self.target = target

    def emit(self, report: Report) -> None:
        payload = asdict(report)
        if report.suppression is not None:
            payload["suppression"]["confirmed_families"] = _families(report.suppression)
        payload["counts"] = dict(Counter(v.status.value for v in report.verdicts))
        with _open(self.target) as out:
            json.dump(payload, out, indent=2, sort_keys=True, default=_json_default)
            out.write("\n")
