"""Report sinks: Markdown and JSON, writing to a path or an open text stream."""

from __future__ import annotations

import json
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import IO, Iterator, Protocol

from benchmark_monitor.model import Status
from benchmark_monitor.report import Report


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


def _cell(text: str) -> str:
    """Make *text* safe inside a Markdown table cell: no pipes, backticks or line breaks."""
    return " ".join(text.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'").split())


class MarkdownSink:
    """A regressions table, then counts per status and warnings."""

    def __init__(self, target: str | Path | IO[str]):
        self.target = target

    def render(self, report: Report) -> str:
        regs = [v for v in report.verdicts if v.status is Status.REGRESSION]
        lines = [f"# Benchmark monitor report for {report.as_of.isoformat()}", "", f"## Regressions ({len(regs)})", ""]
        if regs:
            lines += [
                "| source | platform | tag | benchmark | compiler | metric | params | drop % | z | baseline | n |",
                "|---|---|---|---|---|---|---|---|---|---|---|"
            ]
            for v in sorted(regs, key=lambda v: -(v.drop_pct or 0.0)):
                k = v.key
                cells = tuple(
                    _cell(c)
                    for c in (k.source, k.platform, k.tag, k.benchmark, k.compiler, k.metric, k.params)) + (_fmt(
                        v.drop_pct, ".1f"), _fmt(v.z, ".2f"), _fmt(v.baseline_median, ".4g"), str(v.n_history))
                lines.append("| " + " | ".join(cells) + " |")
        else:
            lines.append("None.")
        counts = Counter(v.status for v in report.verdicts)
        lines += ["", "## Counts", ""] + [f"- {s.value}: {counts.get(s, 0)}" for s in Status]
        if report.warnings:
            lines += ["", "## Warnings", ""] + [f"- {w}" for w in report.warnings]
        return "\n".join(lines) + "\n"

    def emit(self, report: Report) -> None:
        with _open(self.target) as out:
            out.write(self.render(report))


class JsonSink:
    """Full report as JSON (dates ISO-formatted, enums by value)."""

    def __init__(self, target: str | Path | IO[str]):
        self.target = target

    def emit(self, report: Report) -> None:
        payload = json.loads(json.dumps(asdict(report), default=lambda o: o.value if hasattr(o, "value") else str(o)))
        payload["counts"] = dict(Counter(v.status.value for v in report.verdicts))
        with _open(self.target) as out:
            json.dump(payload, out, indent=2, sort_keys=True)
            out.write("\n")
