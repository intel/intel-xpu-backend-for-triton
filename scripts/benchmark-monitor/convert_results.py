"""Parse benchmark report CSVs and append current run data to historical JSON files.

Reads *-report.csv files produced by build_report.py, filters to triton-compiler
results, and appends a single timestamped entry per GPU platform (PVC or BMG) to
the corresponding history.json in the benchmark-data branch checkout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU device string to platform directory mapping
# ---------------------------------------------------------------------------

_GPU_PLATFORM_RULES: list[tuple[list[str], str]] = [
    (["Max 1550", "Max 1100"], "pvc"),
    (["B580", "BMG"], "bmg"),
]


def detect_platform(gpu_device: str) -> str | None:
    """Return the platform directory name ('pvc' or 'bmg') for a gpu_device string.

    Returns None if the device is not recognized.
    """
    for keywords, platform in _GPU_PLATFORM_RULES:
        for kw in keywords:
            if kw in gpu_device:
                return platform
    return None


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------


def load_report_csvs(reports_dir: Path) -> pd.DataFrame:
    """Read and concatenate all *-report.csv files under *reports_dir*.

    Arguments:
        reports_dir: directory containing report CSV files.

    Returns:
        Combined DataFrame, or an empty DataFrame if no files found.
    """
    csv_files = sorted(reports_dir.glob("*-report.csv"))
    if not csv_files:
        logger.warning("No *-report.csv files found in %s", reports_dir)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for csv_file in csv_files:
        logger.info("Reading %s", csv_file)
        df = pd.read_csv(csv_file)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# History entry construction
# ---------------------------------------------------------------------------


def _build_result_key(benchmark: str, compiler: str, params: str) -> str:
    """Construct the result key: ``{benchmark}/{compiler}/{params}``."""
    return f"{benchmark}/{compiler}/{params}"


def build_history_entry(
    df: pd.DataFrame,
    *,
    run_id: str,
    commit_sha: str,
    tag: str,
) -> dict:
    """Build a single history entry dict from a DataFrame of triton rows for one platform.

    Arguments:
        df: DataFrame filtered to compiler=='triton' and a single GPU platform.
        run_id: GitHub Actions run ID.
        commit_sha: git commit SHA.
        tag: run tag string.

    Returns:
        History entry dict ready to be appended to history.json.
    """
    first_row = df.iloc[0]

    entry_datetime = str(first_row.get("datetime", ""))
    agama_version = str(first_row.get("agama_version", ""))
    libigc1_version = str(first_row.get("libigc1_version", ""))

    results: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        key = _build_result_key(
            str(row["benchmark"]),
            str(row["compiler"]),
            str(row["params"]),
        )
        metrics: dict[str, float] = {}

        tflops = row.get("tflops")
        if pd.notna(tflops):
            metrics["tflops"] = float(tflops)

        # TODO: Collect and analyze hbm_gbs (memory bandwidth) metric once
        # detect_regressions.py supports multi-metric analysis.

        if metrics:
            results[key] = metrics

    return {
        "run_id": run_id,
        "datetime": entry_datetime,
        "tag": tag,
        "commit_sha": commit_sha,
        "agama_version": agama_version,
        "libigc1_version": libigc1_version,
        "results": results,
    }


# ---------------------------------------------------------------------------
# History file I/O
# ---------------------------------------------------------------------------


def read_history(path: Path) -> list[dict]:
    """Read an existing history.json or return an empty list if it does not exist."""
    if not path.exists():
        logger.info("History file %s does not exist; starting fresh.", path)
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("History file %s is not a JSON array; starting fresh.", path)
        return []
    return data


def write_history(path: Path, history: list[dict]) -> None:
    """Write the history list back to *path* with readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
        f.write("\n")
    logger.info("Wrote %d entries to %s", len(history), path)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def convert(
    reports_dir: Path,
    history_dir: Path,
    *,
    tag: str,
    run_id: str,
    commit_sha: str,
) -> None:
    """Core conversion: read CSVs, build entries, append to history files.

    Arguments:
        reports_dir: directory with *-report.csv files.
        history_dir: directory containing ``pvc/history.json`` and ``bmg/history.json``.
        tag: run tag.
        run_id: GitHub Actions run ID.
        commit_sha: git commit SHA.
    """
    all_df = load_report_csvs(reports_dir)
    if all_df.empty:
        logger.warning("No data loaded; nothing to convert.")
        return

    # Filter to triton compiler rows only
    if "compiler" not in all_df.columns:
        logger.error("CSV data missing 'compiler' column; cannot filter.")
        return

    triton_df = all_df[all_df["compiler"] == "triton"].copy()
    if triton_df.empty:
        logger.warning("No rows with compiler=='triton' found; nothing to convert.")
        return

    # Detect platform per row
    if "gpu_device" not in triton_df.columns:
        logger.error("CSV data missing 'gpu_device' column; cannot determine platform.")
        return

    triton_df["_platform"] = triton_df["gpu_device"].apply(detect_platform)

    skipped = triton_df[triton_df["_platform"].isna()]
    if not skipped.empty:
        unique_devices = skipped["gpu_device"].unique()
        for dev in unique_devices:
            logger.warning("Skipping unrecognized gpu_device: %s", dev)

    # Process each detected platform
    for platform, group_df in triton_df.groupby("_platform"):
        if platform is None:
            continue

        history_path = history_dir / platform / "history.json"
        history = read_history(history_path)

        # Deduplicate: replace any existing entry with the same run_id.
        history = [h for h in history if h.get("run_id") != run_id]

        entry = build_history_entry(
            group_df,
            run_id=run_id,
            commit_sha=commit_sha,
            tag=tag,
        )
        history.append(entry)

        # Prune old entries to keep history manageable.
        MAX_HISTORY_ENTRIES = 200
        if len(history) > MAX_HISTORY_ENTRIES:
            logger.info(
                "Pruning %s history from %d to %d entries.",
                platform,
                len(history),
                MAX_HISTORY_ENTRIES,
            )
            history = history[-MAX_HISTORY_ENTRIES:]

        write_history(history_path, history)
        logger.info(
            "Appended entry for platform=%s with %d results (run_id=%s).",
            platform,
            len(entry["results"]),
            run_id,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert benchmark report CSVs to historical JSON entries.", )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        required=True,
        help="Directory containing *-report.csv files.",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        required=True,
        help="Directory containing pvc/history.json and bmg/history.json.",
    )
    parser.add_argument(
        "--tag",
        required=True,
        help="Run tag (e.g. 'ci', 'pr-123', 'test').",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="GitHub Actions run ID.",
    )
    parser.add_argument(
        "--commit-sha",
        required=True,
        help="Git commit SHA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args(argv)

    if not args.reports_dir.is_dir():
        logger.error("Reports directory does not exist: %s", args.reports_dir)
        sys.exit(1)

    convert(
        args.reports_dir,
        args.history_dir,
        tag=args.tag,
        run_id=args.run_id,
        commit_sha=args.commit_sha,
    )


if __name__ == "__main__":
    main()
