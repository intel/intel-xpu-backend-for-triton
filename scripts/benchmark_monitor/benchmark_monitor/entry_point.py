"""CLI dispatcher for benchmark-monitor.

Provides four subcommands: convert, detect, report, bootstrap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Self

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

    # detect
    output: Path | None = None
    runner_label: str = ""
    config: Path = Path("scripts/benchmark_monitor/thresholds.yaml")

    # report
    report: str | None = None
    pr_number: str = ""
    run_url: str = ""
    repo: str = "intel/intel-xpu-backend-for-triton"

    # bootstrap
    max_runs: int = 50
    workflow: str = "triton-benchmarks.yml"
    actor: str = "glados-intel"
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
        p = subparsers.add_parser("convert", help="Convert benchmark report CSVs to historical JSON entries.")
        p.add_argument("--reports-dir", type=Path, required=True, help="Directory containing *-report.csv files.")
        p.add_argument(
            "--history-dir",
            type=Path,
            required=True,
            help="Directory containing pvc/history.json and bmg/history.json.",
        )
        p.add_argument("--tag", required=True, help="Run tag (e.g. 'ci', 'pr-123', 'test').")
        p.add_argument("--run-id", required=True, help="GitHub Actions run ID.")
        p.add_argument("--commit-sha", required=True, help="Git commit SHA.")

        # --- detect ---
        p = subparsers.add_parser("detect", help="Detect benchmark performance regressions using Modified Z-Score.")
        p.add_argument("--history-dir", type=Path, required=True, help="Directory containing <gpu>/history.json files")
        p.add_argument("--output", type=Path, required=True, help="Path to write regression-report.json")
        p.add_argument("--runner-label", type=str, default="", help="Runner label (e.g. max1550, b580). Empty = all.")
        p.add_argument(
            "--config",
            type=Path,
            default=Path("scripts/benchmark_monitor/thresholds.yaml"),
            help="Path to thresholds.yaml",
        )

        # --- report ---
        p = subparsers.add_parser("report", help="Post benchmark regression results to GitHub.")
        p.add_argument("--report", required=True, help="Path to regression-report.json")
        p.add_argument("--tag", required=True, help='Run tag (e.g., "ci", "pr-123", "test")')
        p.add_argument("--pr-number", default="", help="PR number (empty string if not a PR run)")
        p.add_argument("--run-url", default="", help="Full URL to the GitHub Actions run")
        p.add_argument("--repo", default="intel/intel-xpu-backend-for-triton", help='Repository in "owner/repo" format')

        # --- bootstrap ---
        p = subparsers.add_parser("bootstrap", help="Bootstrap benchmark history from GitHub Actions artifacts")
        p.add_argument("--history-dir", required=True, help="Path to benchmark-data directory")
        p.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs to download")
        p.add_argument("--workflow", default="triton-benchmarks.yml", help="Workflow filename")
        p.add_argument("--actor", default="glados-intel", help="Filter by actor (default: glados-intel)")
        p.add_argument("--dry-run", action="store_true", help="List runs without downloading")

        return parser

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> Self:
        """Parse CLI arguments into a Config dataclass."""
        parser = cls.build_parser()
        args = parser.parse_args(argv)
        return cls(**vars(args))


@dataclass
class ActionRunner(ABC):
    """Base class for subcommand runners."""

    config: Config

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass
class ConvertActionRunner(ActionRunner):
    """Convert benchmark report CSVs to historical JSON entries."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        from benchmark_monitor.convert_results import convert

        cfg = self.config
        assert cfg.reports_dir is not None and cfg.history_dir is not None
        assert cfg.tag is not None and cfg.run_id is not None and cfg.commit_sha is not None
        if not cfg.reports_dir.is_dir():
            logger.error("Reports directory does not exist: %s", cfg.reports_dir)
            sys.exit(1)

        convert(
            cfg.reports_dir,
            cfg.history_dir,
            tag=cfg.tag,
            run_id=cfg.run_id,
            commit_sha=cfg.commit_sha,
        )


@dataclass
class DetectActionRunner(ActionRunner):
    """Detect benchmark performance regressions using Modified Z-Score."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        from benchmark_monitor.detect_regressions import build_report, load_config

        cfg = self.config
        assert cfg.history_dir is not None and cfg.output is not None
        threshold_config = load_config(cfg.config)
        report = build_report(cfg.history_dir, cfg.runner_label, threshold_config)

        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

        # Print summary to stderr.
        for gpu_name, gpu_data in report.get("gpus", {}).items():
            n_reg = len(gpu_data.get("regressions", []))
            n_imp = len(gpu_data.get("improvements", []))
            print(
                f"[{gpu_name}] checked={gpu_data['total_checked']}  "
                f"regressions={n_reg}  improvements={n_imp}  skipped={gpu_data['skipped']}",
                file=sys.stderr,
            )

        # Exit non-zero if any regressions detected (useful for CI gating).
        total_regressions = sum(len(g.get("regressions", [])) for g in report.get("gpus", {}).values())
        if total_regressions > 0:
            print(f"FAIL: {total_regressions} regression(s) detected.", file=sys.stderr)
            sys.exit(1)


@dataclass
class ReportActionRunner(ActionRunner):
    """Post benchmark regression results to GitHub."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        from benchmark_monitor.report_results import handle_ci, handle_default, handle_pr

        cfg = self.config
        assert cfg.report is not None and cfg.tag is not None

        try:
            with open(cfg.report, encoding="utf-8") as f:
                report = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to read report file %s: %s", cfg.report, exc)
            sys.exit(1)

        match cfg.tag:
            case "ci":
                handle_ci(report, run_url=cfg.run_url, repo=cfg.repo)
            case tag if tag.startswith("pr-") or cfg.pr_number:
                pr_number = cfg.pr_number or tag.removeprefix("pr-")
                if not pr_number:
                    logger.error("PR number could not be determined from --tag or --pr-number.")
                    sys.exit(1)
                handle_pr(report, pr_number=pr_number, repo=cfg.repo)
            case tag:
                handle_default(report, tag=tag)


@dataclass
class BootstrapActionRunner(ActionRunner):
    """Bootstrap benchmark history from GitHub Actions artifacts."""

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        from benchmark_monitor.bootstrap_history import bootstrap

        cfg = self.config
        bootstrap(
            history_dir=str(cfg.history_dir),
            max_runs=cfg.max_runs,
            workflow=cfg.workflow,
            actor=cfg.actor,
            dry_run=cfg.dry_run,
        )


_RUNNERS: dict[str, type[ActionRunner]] = {
    "convert": ConvertActionRunner,
    "detect": DetectActionRunner,
    "report": ReportActionRunner,
    "bootstrap": BootstrapActionRunner,
}


def run(config: Config) -> None:
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
    run(Config.from_args(argv))


if __name__ == "__main__":
    main()
