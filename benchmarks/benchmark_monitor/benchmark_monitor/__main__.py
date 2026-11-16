"""CLI: python -m benchmark_monitor [--catalog PATH] --artifacts DIR --scope NAME [...]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

from benchmark_monitor.aggregate import aggregate
from benchmark_monitor.catalog import CatalogError, load_catalog
from benchmark_monitor.gate import evaluate
from benchmark_monitor.report import Report
from benchmark_monitor.sink import JsonSink, MarkdownSink
from benchmark_monitor.source import ArtifactSource, gated_observations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark_monitor")
    ap.add_argument("--catalog", default=None, help="catalog YAML (default: the shipped default.yaml)")
    ap.add_argument("--artifacts", required=True, help="directory with *-report.csv files (searched recursively)")
    ap.add_argument("--scope", required=True)
    ap.add_argument("--as-of", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--format", choices=("md", "json"), default="md")
    ap.add_argument("--out", default=None, help="output path (default: stdout)")
    args = ap.parse_args(argv)
    try:
        return _run(args)
    except (CatalogError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    if args.scope not in catalog.scopes:
        print(f"unknown scope {args.scope!r}; known: {sorted(catalog.scopes)}", file=sys.stderr)
        return 2
    if not Path(args.artifacts).is_dir():
        print(f"--artifacts {args.artifacts!r} is not a directory", file=sys.stderr)
        return 2
    as_of = args.as_of or datetime.now(timezone.utc).date()
    start = as_of - timedelta(days=catalog.max_window())  # enough history for the baseline
    observations, warnings = gated_observations(
        catalog,
        ArtifactSource(args.artifacts, catalog).observations(catalog.scopes[args.scope], start, as_of))
    by_key = defaultdict(list)
    for obs in observations:
        by_key[obs.key].append(obs)
    series = [aggregate(group) for group in by_key.values()]
    if not series:
        warnings.append(f"no observations for scope {args.scope!r} in {start}..{as_of}")
    report = Report(as_of=as_of, verdicts=tuple(evaluate(s, catalog.resolve(s.key), as_of) for s in series),
                    warnings=tuple(warnings))

    sink_cls = MarkdownSink if args.format == "md" else JsonSink
    sink_cls(args.out or sys.stdout).emit(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
