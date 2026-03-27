"""Layer 2a data source — abstracts where raw benchmark report data comes from.

Provides a uniform interface for fetching benchmark report CSVs, whether from
a local directory or from GitHub Actions artifacts.

Depends on Layer 0 (model.py) only.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ReportSource(ABC):  # pylint: disable=too-few-public-methods
    """Abstract base for benchmark report data sources."""

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch raw benchmark report data and return it as a DataFrame."""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _read_report_csvs(directory: Path) -> pd.DataFrame:
    """Read and concatenate all ``*-report.csv`` files under *directory*."""
    csv_files = sorted(directory.glob("*-report.csv"))
    if not csv_files:
        logger.warning("No *-report.csv files found in %s", directory)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.info("Reading %s", csv_file)
        df = pd.read_csv(csv_file)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Concrete sources
# ---------------------------------------------------------------------------


class LocalCSVSource(ReportSource):  # pylint: disable=too-few-public-methods
    """Read benchmark reports from a local directory of CSV files.

    Arguments:
        reports_dir: path to a directory containing ``*-report.csv`` files.
    """

    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir

    def fetch(self) -> pd.DataFrame:
        """Read and concatenate all ``*-report.csv`` files from the directory.

        Returns:
            Combined DataFrame, or an empty DataFrame if no files are found.
        """
        return _read_report_csvs(self.reports_dir)


class GHArtifactSource(ReportSource):  # pylint: disable=too-few-public-methods
    """Download benchmark reports from a GitHub Actions run artifact.

    Arguments:
        run_id: numeric GitHub Actions run ID.
        artifact_name: name of the artifact to download.
        repo: GitHub repository in ``owner/name`` format.
    """

    def __init__(
        self,
        run_id: int,
        artifact_name: str = "benchmark-reports",
        repo: str = "intel/intel-xpu-backend-for-triton",
    ) -> None:
        self.run_id = run_id
        self.artifact_name = artifact_name
        self.repo = repo

    def fetch(self) -> pd.DataFrame:
        """Download the artifact and read CSVs from it.

        Uses ``gh run download`` to fetch the artifact into a temporary
        directory, then reads all ``*-report.csv`` files found there.

        Returns:
            Combined DataFrame, or an empty DataFrame if the download fails
            or no CSV files are found.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "reports"
            result = self._run_gh([
                "run",
                "download",
                str(self.run_id),
                "--repo",
                self.repo,
                "--name",
                self.artifact_name,
                "--dir",
                str(dest),
            ])

            if result.returncode != 0:
                logger.warning(
                    "Failed to download artifact %r for run %s",
                    self.artifact_name,
                    self.run_id,
                )
                return pd.DataFrame()

            return _read_report_csvs(dest)

    @staticmethod
    def _run_gh(args: list[str]) -> subprocess.CompletedProcess:
        """Run a ``gh`` CLI command and return the result.

        Logs to stderr on failure but never raises.
        """
        cmd = ["gh"] + args
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            logger.warning("gh command failed: %s", " ".join(cmd))
            if result.stderr:
                logger.warning("  stderr: %s", result.stderr.strip())
        return result
