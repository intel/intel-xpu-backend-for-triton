"""CLI: python -m benchmark_monitor [--catalog PATH] --artifacts DIR --scope NAME [...]."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import date, datetime, timezone

from benchmark_monitor.catalog import CatalogError, load_catalog
from benchmark_monitor.pipeline import run
from benchmark_monitor.sink import JsonSink, MarkdownSink
from benchmark_monitor.source import ArtifactSource


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark_monitor")
    ap.add_argument("--catalog", default=None, help="catalog YAML (default: the shipped default.yaml)")
    ap.add_argument("--artifacts", required=True, help="directory with *-report.csv files (searched recursively)")
    ap.add_argument("--scope", required=True)
    ap.add_argument("--as-of", type=date.fromisoformat, default=None, help="YYYY-MM-DD (default: today, UTC)")
    ap.add_argument("--nights", type=int, default=30,
                    help="nights of verdict history to evaluate and suppress over (>= 1)")
    ap.add_argument(
        "--history-days", type=int, default=None, help=
        "calendar days of history to read before the first evaluated night (default: 3 x the largest catalog window; "
        "0 = all history). The gate then takes the last `window` points per series")
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
    if args.nights < 1:
        print(f"--nights must be at least 1, got {args.nights}", file=sys.stderr)
        return 2
    as_of = args.as_of or datetime.now(timezone.utc).date()
    report = run(catalog, ArtifactSource(args.artifacts, catalog), [args.scope], as_of, args.nights,
                 history_days=args.history_days)

    sink_cls = MarkdownSink if args.format == "md" else JsonSink
    sink_cls(args.out or sys.stdout).emit(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
