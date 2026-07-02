"""Layer 3 ingestion/ETL — pure data transformation from DataFrames to domain objects.

Takes raw benchmark report DataFrames and produces structured ``BenchmarkEntry``
instances grouped by GPU platform.

Depends on Layer 0 (model.py) only.
"""

from __future__ import annotations

import logging

import pandas as pd

from benchmark_monitor.model import BenchmarkEntry, detect_platform

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Key construction
# ---------------------------------------------------------------------------


def _build_result_key(benchmark: str, compiler: str, params: str) -> str:
    """Construct the result key: ``{benchmark}/{compiler}/{params}``."""
    return f"{benchmark}/{compiler}/{params}"


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------


def parse_reports(  # pylint: disable=too-many-locals
    df: pd.DataFrame,
    *,
    run_id: str,
    commit_sha: str,
    tag: str,
) -> dict[str, BenchmarkEntry]:
    """Parse a raw report DataFrame into per-platform ``BenchmarkEntry`` objects.

    Filters to ``compiler == "triton"`` rows, groups by detected GPU platform,
    and builds a ``BenchmarkEntry`` for each platform.

    Arguments:
        df: raw benchmark report DataFrame (as returned by a ``ReportSource``).
        run_id: GitHub Actions run ID.
        commit_sha: git commit SHA for the run.
        tag: run tag string (e.g. ``"ci"``).

    Returns:
        Dict mapping GPU platform string (``"pvc"``, ``"bmg"``) to the
        corresponding ``BenchmarkEntry``.  Empty dict if no usable data is
        found.
    """
    if df.empty:
        logger.warning("Empty DataFrame; nothing to parse.")
        return {}

    # --- validate required columns ---
    if "compiler" not in df.columns:
        logger.warning("DataFrame missing 'compiler' column; cannot filter.")
        return {}

    triton_df = df[df["compiler"] == "triton"].copy()
    if triton_df.empty:
        logger.warning("No rows with compiler=='triton' found.")
        return {}

    if "gpu_device" not in triton_df.columns:
        logger.warning("DataFrame missing 'gpu_device' column; cannot determine platform.")
        return {}

    # --- detect platform per row ---
    triton_df["_platform"] = triton_df["gpu_device"].apply(detect_platform)

    skipped = triton_df[triton_df["_platform"].isna()]
    if not skipped.empty:
        for dev in skipped["gpu_device"].unique():
            logger.warning("Skipping unrecognized gpu_device: %s", dev)

    # --- build one BenchmarkEntry per platform ---
    entries: dict[str, BenchmarkEntry] = {}

    for platform, group_df in triton_df.groupby("_platform"):
        if platform is None:
            continue

        first_row = group_df.iloc[0]
        entry_datetime = str(first_row.get("datetime", ""))
        agama_version = str(first_row.get("agama_version", ""))
        libigc1_version = str(first_row.get("libigc1_version", ""))

        results: dict[str, dict[str, float]] = {}
        for _, row in group_df.iterrows():
            key = _build_result_key(
                str(row["benchmark"]),
                str(row["compiler"]),
                str(row["params"]),
            )
            metrics: dict[str, float] = {}

            tflops = row.get("tflops")
            if pd.notna(tflops):
                metrics["tflops"] = float(tflops)

            # TODO: Collect and analyze hbm_gbs (memory bandwidth) metric once  # pylint: disable=fixme
            # detect_regressions.py supports multi-metric analysis.

            if metrics:
                results[key] = metrics

        entries[str(platform)] = BenchmarkEntry(
            run_id=run_id,
            datetime=entry_datetime,
            tag=tag,
            commit_sha=commit_sha,
            agama_version=agama_version,
            libigc1_version=libigc1_version,
            results=results,
        )

    return entries
