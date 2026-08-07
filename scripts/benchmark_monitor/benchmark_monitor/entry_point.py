"""CLI dispatcher for benchmark-monitor.

Provides three subcommands: convert, detect, bootstrap.
Wires together the layered architecture (source → ingest → storage → analysis → formatting).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from benchmark_monitor.analysis import analyze
from benchmark_monitor.formatting import (
    build_issue_body,
    build_pr_comment,
    format_issue_title,
    format_summary,
    improvement_table,
    regression_table,
    sort_improvements,
    sort_regressions,
)
from benchmark_monitor.ingest import parse_reports
from benchmark_monitor.model import RUNNER_TO_GPU, DetectionConfig, MetricResult, ThresholdConfig
from benchmark_monitor.source import GHArtifactSource, LocalCSVSource
from benchmark_monitor.storage import JsonFileBackend, ParquetBackend

logger = logging.getLogger(__name__)


@dataclass
class Config:  # pylint: disable=too-many-instance-attributes
    """Holds all CLI arguments across subcommands."""

    action: str | None = None

    # convert
    reports_dir: Path | None = None
    history_dir: Path | None = None
    tag: str | None = None
    run_id: str | None = None
    commit_sha: str | None = None
    backend: str = "json"

    # detect
    output: Path | None = None
    runner_label: str = ""
    config: Path = Path("scripts/benchmark_monitor/thresholds.yaml")
    format: str = "json"
    gpu: str = ""
    run_url: str = ""

    # bootstrap
    max_runs: int = 50
    workflow: str = "triton-benchmarks.yml"
    actor: str = "glados-intel"
    repo: str = "intel/intel-xpu-backend-for-triton"
    dry_run: bool = False

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Build the full argparse parser with subparsers for all subcommands."""
        parser = argparse.ArgumentParser(
            prog="benchmark-monitor",
            description="Benchmark regression detection for Triton CI.",
        )
        subparsers = parser.add_subparsers(dest="action", required=True)

        # --- convert ---
        p = subparsers.add_parser("convert", help="Convert benchmark report CSVs to historical entries.")
        p.add_argument("--reports-dir", type=Path, required=True, help="Directory containing *-report.csv files.")
        p.add_argument(
            "--history-dir",
            type=Path,
            required=True,
            help="Directory containing pvc/ and bmg/ history data.",
        )
        p.add_argument("--tag", required=True, help="Run tag (e.g. 'ci', 'pr-123', 'test').")
        p.add_argument("--run-id", required=True, help="GitHub Actions run ID.")
        p.add_argument("--commit-sha", required=True, help="Git commit SHA.")
        _add_backend_arg(p)

        # --- detect ---
        p = subparsers.add_parser("detect", help="Detect benchmark performance regressions using Modified Z-Score.")
        p.add_argument("--history-dir", type=Path, required=True, help="Directory containing <gpu>/ history data.")
        p.add_argument("--output", type=Path, required=True, help="Path to write regression-report.json")
        p.add_argument("--runner-label", type=str, default="", help="Runner label (e.g. max1550, b580). Empty = all.")
        p.add_argument(
            "--config",
            type=Path,
            default=Path("scripts/benchmark_monitor/thresholds.yaml"),
            help="Path to thresholds.yaml",
        )
        _add_backend_arg(p)
        p.add_argument(
            "--format",
            choices=["json", "markdown", "text", "issue-title", "issue-body", "pr-comment"],
            default="json",
            help=("Output format (default: json). "
                  "'issue-title' and 'issue-body' produce content for GitHub issue creation. "
                  "'pr-comment' produces a PR comment body with marker. "
                  "Use with --gpu to target a specific GPU for issue formats."),
        )
        p.add_argument(
            "--gpu",
            type=str,
            default="",
            help="GPU platform for issue-title/issue-body formats (e.g. pvc, bmg).",
        )
        p.add_argument(
            "--run-url",
            type=str,
            default="",
            help="GitHub Actions run URL (used in issue-body format).",
        )

        # --- bootstrap ---
        p = subparsers.add_parser("bootstrap", help="Bootstrap benchmark history from GitHub Actions artifacts")
        p.add_argument("--history-dir", type=Path, required=True, help="Path to benchmark-data directory")
        p.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs to download")
        p.add_argument("--workflow", default="triton-benchmarks.yml", help="Workflow filename")
        p.add_argument("--actor", default="glados-intel", help="Filter by actor (default: glados-intel)")
        p.add_argument(
            "--repo",
            default="intel/intel-xpu-backend-for-triton",
            help='Repository in "owner/repo" format',
        )
        p.add_argument("--dry-run", action="store_true", help="List runs without downloading")
        _add_backend_arg(p)

        return parser

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Config:
        """Parse CLI arguments into a Config dataclass."""
        parser = cls.build_parser()
        args = parser.parse_args(argv)
        return cls(**vars(args))


def _add_backend_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --backend argument to a subparser."""
    parser.add_argument(
        "--backend",
        choices=["json", "parquet", "db"],
        default="json",
        help="History storage backend (default: json).",
    )


# ---------------------------------------------------------------------------
# Storage backend factory
# ---------------------------------------------------------------------------


def _make_backend(backend_name: str, history_dir: Path) -> JsonFileBackend | ParquetBackend:
    """Instantiate the appropriate HistoryBackend."""
    if backend_name == "json":
        return JsonFileBackend(history_dir)
    if backend_name == "parquet":
        return ParquetBackend(history_dir)
    if backend_name == "db":
        raise SystemExit("ERROR: Database backend is planned for a future release. Use --backend json or parquet.")
    raise ValueError(f"Unknown backend: {backend_name}")


# ---------------------------------------------------------------------------
# Config loading helper
# ---------------------------------------------------------------------------


def _load_detection_config(path: Path) -> DetectionConfig:
    """Load thresholds.yaml and return a DetectionConfig."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    defaults_raw = raw.get("defaults", {})
    defaults = ThresholdConfig(
        min_history=defaults_raw.get("min_history", 8),
        rolling_window=defaults_raw.get("rolling_window", 20),
        z_threshold=defaults_raw.get("z_threshold", 3.0),
        min_drop_pct=defaults_raw.get("min_drop_pct", 5.0),
        improvement_lock_pct=defaults_raw.get("improvement_lock_pct", 8.0),
        max_cv=defaults_raw.get("max_cv", 0.15),
    )

    overrides: dict[str, ThresholdConfig] = {}
    for name, vals in raw.get("overrides", {}).items():
        overrides[name] = ThresholdConfig(
            min_history=vals.get("min_history", defaults.min_history),
            rolling_window=vals.get("rolling_window", defaults.rolling_window),
            z_threshold=vals.get("z_threshold", defaults.z_threshold),
            min_drop_pct=vals.get("min_drop_pct", defaults.min_drop_pct),
            improvement_lock_pct=vals.get("improvement_lock_pct", defaults.improvement_lock_pct),
            max_cv=vals.get("max_cv", defaults.max_cv),
        )

    return DetectionConfig(defaults=defaults, overrides=overrides)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metric_to_dict(metric: MetricResult, change_key: str = "drop_pct") -> dict:
    """Convert a MetricResult to a dict matching the original JSON schema."""
    return {
        "key": metric.key,
        "benchmark": metric.benchmark,
        "params": metric.params,
        "current_tflops": metric.current_tflops,
        "baseline_median": metric.baseline_median,
        change_key: metric.change_pct,
        "modified_z": metric.modified_z,
        "driver_change": metric.driver_change,
    }


def _emit_formatted_output(  # pylint: disable=too-many-branches
    fmt: str,
    report: dict,
    gpus_output: dict[str, Any],
    cfg: Config,
    latest_tag: str,
) -> None:
    """Write formatted output to stdout based on --format flag."""
    if fmt == "json":
        for gpu_name, gpu_data in gpus_output.items():
            n_reg = len(gpu_data.get("regressions", []))
            n_imp = len(gpu_data.get("improvements", []))
            print(
                f"[{gpu_name}] checked={gpu_data['total_checked']}  "
                f"regressions={n_reg}  improvements={n_imp}  skipped={gpu_data['skipped']}",
                file=sys.stderr,
            )
    elif fmt == "text":
        print(format_summary(report, tag=latest_tag))
    elif fmt == "markdown":
        for _gpu_name, gpu_data in gpus_output.items():
            regs = sort_regressions(gpu_data.get("regressions", []))
            imps = sort_improvements(gpu_data.get("improvements", []))
            if regs:
                print(regression_table(regs))
            if imps:
                print(improvement_table(imps))
    elif fmt == "issue-title":
        target_gpu = cfg.gpu or next(iter(gpus_output), "")
        if target_gpu and target_gpu in gpus_output:
            gpu_data = gpus_output[target_gpu]
            print(format_issue_title(target_gpu, len(gpu_data.get("regressions", [])), gpu_data.get("driver_change")))
    elif fmt == "issue-body":
        target_gpu = cfg.gpu or next(iter(gpus_output), "")
        if target_gpu and target_gpu in gpus_output:
            print(
                build_issue_body(
                    target_gpu,
                    gpus_output[target_gpu],
                    run_id=report.get("run_id", ""),
                    run_url=cfg.run_url,
                    commit_sha=report.get("commit_sha", ""),
                    datetime_str=report.get("datetime", ""),
                ))
    elif fmt == "pr-comment":
        print(build_pr_comment(report))


# ---------------------------------------------------------------------------
# Action runners
# ---------------------------------------------------------------------------


@dataclass
class ActionRunner(ABC):
    """Base class for subcommand runners."""

    config: Config

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass
class ConvertActionRunner(ActionRunner):
    """source → ingest → storage."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        cfg = self.config
        assert cfg.reports_dir is not None and cfg.history_dir is not None
        assert cfg.tag is not None and cfg.run_id is not None and cfg.commit_sha is not None
        if not cfg.reports_dir.is_dir():
            logger.error("Reports directory does not exist: %s", cfg.reports_dir)
            sys.exit(1)

        backend = _make_backend(cfg.backend, cfg.history_dir)
        source = LocalCSVSource(cfg.reports_dir)
        df = source.fetch()
        if df.empty:
            logger.warning("No data loaded; nothing to convert.")
            return

        entries = parse_reports(df, run_id=cfg.run_id, commit_sha=cfg.commit_sha, tag=cfg.tag)
        for gpu, entry in entries.items():
            history = backend.load(gpu)
            history.entries.append(entry)
            backend.save(gpu, history)
            logger.info("Appended entry for platform=%s with %d results (run_id=%s).", gpu, len(entry.results),
                        cfg.run_id)


@dataclass
class DetectActionRunner(ActionRunner):
    """storage → analysis → formatting."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:  # pylint: disable=too-many-locals
        cfg = self.config
        assert cfg.history_dir is not None and cfg.output is not None
        detection_config = _load_detection_config(cfg.config)
        backend = _make_backend(cfg.backend, cfg.history_dir)

        gpu_list = self._resolve_gpu_list(cfg, backend)
        gpus_output, latest_entry = self._analyze_gpus(gpu_list, backend, detection_config)
        report = self._build_report(gpus_output, latest_entry)

        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

        _emit_formatted_output(cfg.format, report, gpus_output, cfg, latest_entry.tag if latest_entry else "unknown")

        total_regressions = sum(len(g.get("regressions", [])) for g in gpus_output.values())
        if total_regressions > 0:
            print(f"FAIL: {total_regressions} regression(s) detected.", file=sys.stderr)
            sys.exit(1)

    @staticmethod
    def _resolve_gpu_list(cfg: Config, backend: JsonFileBackend | ParquetBackend) -> list[str]:
        if cfg.runner_label:
            gpu = RUNNER_TO_GPU.get(cfg.runner_label)
            if gpu is None:
                logger.warning("Unknown runner label '%s', scanning all GPUs.", cfg.runner_label)
                return backend.list_gpus()
            return [gpu]
        return backend.list_gpus()

    @staticmethod
    def _analyze_gpus(
        gpu_list: list[str],
        backend: JsonFileBackend | ParquetBackend,
        detection_config: DetectionConfig,
    ) -> tuple[dict[str, Any], Any]:
        latest_entry = None
        gpus_output: dict[str, Any] = {}
        for gpu_name in gpu_list:
            history = backend.load(gpu_name)
            if not history.entries:
                continue
            current_entry = history.entries[-1]
            if latest_entry is None or current_entry.datetime > latest_entry.datetime:
                latest_entry = current_entry
            result = analyze(history, detection_config)
            gpus_output[gpu_name] = {
                "regressions": [_metric_to_dict(m, "drop_pct") for m in result.regressions],
                "improvements": [_metric_to_dict(m, "gain_pct") for m in result.improvements],
                "skipped": result.skipped,
                "total_checked": result.total_checked,
                "driver_change": result.driver_change,
            }
        return gpus_output, latest_entry

    @staticmethod
    def _build_report(gpus_output: dict[str, Any], latest_entry: Any) -> dict:
        return {
            "run_id": latest_entry.run_id if latest_entry else "",
            "datetime": latest_entry.datetime if latest_entry else "",
            "commit_sha": latest_entry.commit_sha if latest_entry else "",
            "gpus": gpus_output,
        }


@dataclass
class BootstrapActionRunner(ActionRunner):  # pylint: disable=too-many-instance-attributes
    """GHArtifactSource (loop) → ingest → storage."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        cfg = self.config
        assert cfg.history_dir is not None
        backend = _make_backend(cfg.backend, cfg.history_dir)

        print(f"Listing up to {cfg.max_runs} successful runs for {cfg.workflow}...")
        runs = self._list_runs(cfg)
        if runs is None:
            return

        if cfg.dry_run:
            for ci_run in runs:
                print(f"  Run {ci_run['databaseId']} | {ci_run['createdAt']}"
                      f" | {ci_run['headSha'][:8]} | {ci_run['displayTitle']}")
            return

        self._import_runs(runs, cfg, backend)

    @staticmethod
    def _list_runs(cfg: Config) -> list[dict] | None:
        gh_args = [
            "run",
            "list",
            "--repo",
            cfg.repo,
            "--workflow",
            cfg.workflow,
            "--status",
            "success",
            "--limit",
            str(cfg.max_runs),
            "--json",
            "databaseId,createdAt,headSha,displayTitle",
        ]
        if cfg.actor:
            gh_args.extend(["--user", cfg.actor])
        result = GHArtifactSource._run_gh(gh_args)  # pylint: disable=protected-access
        if result.returncode != 0:
            print("Failed to list runs.")
            return None
        runs = json.loads(result.stdout)
        if not runs:
            print("No runs found.")
            return None
        print(f"Found {len(runs)} runs.")
        return runs

    @staticmethod
    def _import_runs(
        runs: list[dict],
        cfg: Config,
        backend: JsonFileBackend | ParquetBackend,
    ) -> None:
        existing_run_ids: set[str] = set()
        for gpu in backend.list_gpus():
            for entry in backend.load(gpu).entries:
                existing_run_ids.add(entry.run_id)

        imported = 0
        skipped_count = 0

        for ci_run in reversed(runs):
            run_id = ci_run["databaseId"]
            if str(run_id) in existing_run_ids:
                print(f"  Run {run_id}: already in history, skipping.")
                skipped_count += 1
                continue

            print(f"  Run {run_id} ({ci_run['createdAt'][:10]}): downloading...", end=" ", flush=True)
            source = GHArtifactSource(run_id=run_id, repo=cfg.repo)
            df = source.fetch()
            if df.empty:
                print("download failed or no CSVs, skipping.")
                continue

            entries = parse_reports(df, run_id=str(run_id), commit_sha=ci_run.get("headSha", ""), tag="ci")
            if not entries:
                print("no triton results found, skipping.")
                continue

            for gpu, entry in entries.items():
                history = backend.load(gpu)
                history.entries.append(entry)
                backend.save(gpu, history)

            existing_run_ids.add(str(run_id))
            imported += 1
            print(f"imported {sum(len(e.results) for e in entries.values())} metrics"
                  f" for {', '.join(g.upper() for g in entries)}.")

        print(f"\nDone. Imported {imported} runs, skipped {skipped_count} (already present).")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, type[ActionRunner]] = {
    "convert": ConvertActionRunner,
    "detect": DetectActionRunner,
    "bootstrap": BootstrapActionRunner,
}


def dispatch(config: Config) -> None:
    """Dispatch to the appropriate ActionRunner based on config.action."""
    if not config.action:
        raise ValueError("No action specified")
    runner_cls = _RUNNERS.get(config.action)
    if runner_cls is None:
        raise ValueError(f"Unknown action: {config.action}")
    runner_cls(config=config)()


def main(argv: list[str] | None = None) -> None:
    """Main CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dispatch(Config.from_args(argv))


if __name__ == "__main__":
    main()
