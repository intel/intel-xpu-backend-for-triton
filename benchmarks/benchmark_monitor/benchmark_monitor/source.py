"""Observation sources: the Source protocol and CSV report readers (long and wide formats)."""

from __future__ import annotations

import csv
import fnmatch
import json
import math
from collections.abc import Iterable, Iterator, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Protocol

from benchmark_monitor.aggregate import utc_night
from benchmark_monitor.catalog import Catalog, Scope
from benchmark_monitor.model import Observation, SeriesKey, canonical_params

# Version and provenance columns copied into Observation.attrs (empty values dropped).
ATTR_COLUMNS = ("libigc1_version", "level_zero_version", "gpu_device", "agama_version", "torch_version",
                "compiler_version", "benchmarking_method", "triton_commit", "gh_run_id")
# Wide-format value columns -> metric name.
WIDE_METRICS = {"tflops": "tflops", "hbm_gbs": "gbps"}


class Source(Protocol):
    """Anything that yields observations for a scope within [start, end] (UTC dates)."""

    def observations(self, scope: Scope, start: date, end: date) -> Iterable[Observation]:
        ...


def _parse_ts(text: str) -> datetime | None:
    """Parse a report timestamp; empty, zero and unparsable values give None."""
    text = (text or "").strip()
    if not text or text.strip("0-: .T") == "":
        return None
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return None
    return moment if moment.year >= 2000 else None


def _finite(text: str) -> float | None:
    """The finite float in *text*; empty, unparsable, nan and inf give None."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _params(text: str) -> str:
    try:
        parsed = json.loads(text) if text else {}
    except ValueError:
        return text
    return canonical_params(parsed) if isinstance(parsed, dict) else text


def _any_glob(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


class CsvReportSource:
    """Reads ``*-report.csv`` files, auto-detecting the format per file.

    long: ``value_name``/``value`` rows from transform_results.py (kind "third_party").
    wide: ``tflops``/``hbm_gbs`` columns from build_report.py, melted to metrics tflops/gbps (kind "micro").
    A scope with source "artifact" accepts both kinds. SeriesKey.source is the kind of the file the row came
    from ("micro" or "third_party"), so long and wide series never merge under one key.
    """

    def __init__(self, paths: Iterable[str | Path], catalog: Catalog):
        self.paths = tuple(Path(p) for p in paths)
        self.catalog = catalog

    def observations(self, scope: Scope, start: date, end: date) -> Iterator[Observation]:
        for path in self.paths:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or ())
                if {"value_name", "value"} <= fields:
                    kind = "third_party"
                elif fields & set(WIDE_METRICS):
                    kind = "micro"
                else:
                    continue
                if scope.source not in (kind, "artifact"):
                    continue
                for row in reader:
                    yield from self._row(row, kind, scope, start, end)

    def _row(self, row: dict[str, str], kind: str, scope: Scope, start: date, end: date) -> Iterator[Observation]:
        moment = _parse_ts(row.get("ts") or row.get("datetime") or "")
        if moment is None or not start <= utc_night(moment) <= end:
            return
        benchmark = row.get("benchmark") or ""
        if scope.tags and (row.get("tag") or "") not in scope.tags:
            return
        if not _any_glob(benchmark, scope.benchmarks):
            return
        if scope.groups and not _any_glob(row.get("benchmark_group") or "", scope.groups):
            return
        if self.catalog.normalize_platform(row.get("gpu_device") or "") != scope.platform:
            return
        if kind == "third_party":
            values = [(row.get("value_name") or "", _finite(row.get("value", "")))]
        else:
            values = [(metric, _finite(row.get(col, ""))) for col, metric in WIDE_METRICS.items()]
        attrs = {c: row[c] for c in ATTR_COLUMNS if row.get(c)}
        for metric, value in values:
            # Reports write 0 for a metric a row did not measure, so <= 0 is a placeholder unless the
            # catalog declares the metric signed (a difference of two timings can be zero or negative).
            if value is None or not metric or (value <= 0 and not self.catalog.is_signed(metric)):
                continue
            key = SeriesKey(kind, scope.platform,
                            row.get("tag") or "", benchmark,
                            row.get("compiler") or "", metric, _params(row.get("params") or ""))
            yield Observation(key, row.get("run_uuid") or "", moment, value, dict(attrs))


class ArtifactSource:
    """A directory of unzipped CI artifacts: every ``*-report.csv`` below it, read as CSV reports."""

    def __init__(self, directory: str | Path, catalog: Catalog):
        self.directory = Path(directory)
        self.catalog = catalog

    def observations(self, scope: Scope, start: date, end: date) -> Iterator[Observation]:
        paths = sorted(self.directory.rglob("*-report.csv"))
        return CsvReportSource(paths, self.catalog).observations(scope, start, end)


def gated_observations(catalog: Catalog, observations: Iterable[Observation]) -> tuple[list[Observation], list[str]]:
    """Keep observations of gated metrics; one warning per undeclared metric (declared-ungated ones are dropped)."""
    kept: list[Observation] = []
    undeclared: set[str] = set()
    for obs in observations:
        state = catalog.metric_state(obs.key.metric)
        if state == "gated":
            kept.append(obs)
        elif state == "undeclared":
            undeclared.add(obs.key.metric)
    return kept, [f"metric {m!r} is not declared in the catalog; its series were skipped" for m in sorted(undeclared)]
