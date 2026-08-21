"""Layer 5 presentation/formatting — pure string generation, no I/O.

All functions in this module produce formatted strings (markdown tables,
issue bodies, PR comments, summaries).  They depend only on ``model.py``
(Layer 0) for shared constants and never perform file, network, or
subprocess operations.
"""

from __future__ import annotations

import json

from benchmark_monitor.model import GPU_SHORT_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_MONITOR_MARKER = "<!-- benchmark-monitor -->"
MAX_PR_TABLE_ROWS = 20

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def format_params(params_json: str) -> str:
    """Parse a JSON params string and return a compact 'K1=V1, K2=V2' representation."""
    try:
        obj = json.loads(params_json)
        return ", ".join(f"{k}={v}" for k, v in obj.items())
    except (json.JSONDecodeError, TypeError):
        return params_json


def sort_regressions(items: list[dict]) -> list[dict]:
    """Sort regressions by drop_pct ascending (worst first, most negative first)."""
    return sorted(items, key=lambda r: r.get("drop_pct", 0))


def sort_improvements(items: list[dict]) -> list[dict]:
    """Sort improvements by gain_pct descending (best improvement first, most positive first)."""
    return sorted(items, key=lambda r: r.get("gain_pct", 0), reverse=True)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _regression_table_row(item: dict, *, current_label: str = "Current") -> str:  # noqa: ARG001  # pylint: disable=unused-argument
    params = format_params(item.get("params", ""))
    baseline = item.get("baseline_median", 0)
    current = item.get("current_tflops", 0)
    drop = item.get("drop_pct", 0)
    z_score = item.get("modified_z", 0)
    return f"| {item.get('benchmark', '')} | {params} | {baseline:.1f} | {current:.1f} | {drop:+.1f}% | {z_score:.1f} |"


def regression_table(items: list[dict], *, current_label: str = "Current", limit: int = 0) -> str:
    """Build a markdown table for regressions.

    Arguments:
        items: sorted regression dicts.
        current_label: column header for the current value (e.g. "Current" or "PR").
        limit: max rows to show (0 = unlimited).

    Returns:
        Markdown table string.
    """
    header = (f"| Benchmark | Params | Baseline (TFlops) | {current_label} (TFlops) | Change | Z-Score |\n"
              "|-----------|--------|-------------------|" + "-" * (len(current_label) + len(" (TFlops) ")) +
              "|--------|---------|")
    rows = [_regression_table_row(r, current_label=current_label) for r in (items[:limit] if limit else items)]
    table = header + "\n" + "\n".join(rows)
    if limit and len(items) > limit:
        table += f"\n\n... and {len(items) - limit} more"
    return table


def _improvement_table_row(item: dict, *, current_label: str = "Current") -> str:  # noqa: ARG001  # pylint: disable=unused-argument
    params = format_params(item.get("params", ""))
    baseline = item.get("baseline_median", 0)
    current = item.get("current_tflops", 0)
    gain = item.get("gain_pct", 0)
    z_score = item.get("modified_z", 0)
    return f"| {item.get('benchmark', '')} | {params} | {baseline:.1f} | {current:.1f} | {gain:+.1f}% | {z_score:.1f} |"


def improvement_table(items: list[dict], *, current_label: str = "Current", limit: int = 0) -> str:
    """Build a markdown table for improvements."""
    header = (f"| Benchmark | Params | Baseline (TFlops) | {current_label} (TFlops) | Change | Z-Score |\n"
              "|-----------|--------|-------------------|" + "-" * (len(current_label) + len(" (TFlops) ")) +
              "|--------|---------|")
    rows = [_improvement_table_row(r, current_label=current_label) for r in (items[:limit] if limit else items)]
    table = header + "\n" + "\n".join(rows)
    if limit and len(items) > limit:
        table += f"\n\n... and {len(items) - limit} more"
    return table


# ---------------------------------------------------------------------------
# Composite formatters
# ---------------------------------------------------------------------------


def driver_change_notice(driver_change: list[dict] | None) -> str:
    """Return a markdown note about a driver version change, or empty string."""
    if not driver_change:
        return ""
    parts = [f"{c.get('field', '?')}: {c.get('from', '?')} \u2192 {c.get('to', '?')}" for c in driver_change]
    return (f"\n> **Note:** This regression coincides with a driver version change "
            f"({', '.join(parts)}) and may be driver-caused.\n")


def format_issue_title(
    gpu_key: str,
    n_regressions: int,
    driver_change: list[dict] | None = None,
) -> str:
    """Generate a GitHub issue title for a regression report."""
    gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
    if driver_change:
        return f"[Perf Regression - Driver Change] {n_regressions} benchmarks regressed on {gpu_name}"
    return f"[Perf Regression] {n_regressions} benchmarks regressed on {gpu_name}"


def build_issue_body(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    gpu_key: str,
    gpu_data: dict,
    run_id: str,
    run_url: str,
    commit_sha: str,
    datetime_str: str,
) -> str:
    """Build the full markdown body for a GitHub regression issue."""
    regressions = sort_regressions(gpu_data.get("regressions", []))
    improvements = sort_improvements(gpu_data.get("improvements", []))
    total_checked = gpu_data.get("total_checked", 0)
    skipped = gpu_data.get("skipped", 0)
    driver_change = gpu_data.get("driver_change")

    gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())

    body_parts = [
        "## Performance Regression Detected\n",
        f"**Run:** [#{run_id}]({run_url})",
        f"**Commit:** {commit_sha}",
        f"**Date:** {datetime_str}",
        f"**GPU:** {gpu_name}",
        driver_change_notice(driver_change),
        f"### Regressions ({len(regressions)} found, {total_checked} checked, {skipped} skipped)\n",
        regression_table(regressions),
    ]

    if improvements:
        body_parts.append(f"\n### Improvements ({len(improvements)} found)\n")
        body_parts.append(improvement_table(improvements))

    return "\n".join(body_parts)


def build_pr_comment(report: dict) -> str:
    """Build the full PR comment body covering all GPUs."""
    gpus = report.get("gpus", {})
    sections: list[str] = [BENCHMARK_MONITOR_MARKER, "## Benchmark Results\n"]

    for gpu_key, gpu_data in gpus.items():
        gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
        total_checked = gpu_data.get("total_checked", 0)
        skipped = gpu_data.get("skipped", 0)
        regressions = sort_regressions(gpu_data.get("regressions", []))
        improvements = sort_improvements(gpu_data.get("improvements", []))

        sections.append(f"### {gpu_name} ({total_checked} benchmarks, {skipped} skipped)\n")

        if not regressions and not improvements:
            sections.append(":white_check_mark: No significant performance changes detected\n")
            continue

        if regressions:
            table = regression_table(regressions, current_label="PR", limit=MAX_PR_TABLE_ROWS)
            sections.append(f"<details><summary>:red_circle: Regressions ({len(regressions)} found)</summary>\n")
            sections.append(table)
            sections.append("\n</details>\n")

        if improvements:
            table = improvement_table(improvements, current_label="PR", limit=MAX_PR_TABLE_ROWS)
            sections.append(f"<details><summary>:green_circle: Improvements ({len(improvements)} found)</summary>\n")
            sections.append(table)
            sections.append("\n</details>\n")

    return "\n".join(sections)


def format_summary(report: dict, tag: str) -> str:
    """Return a formatted plain-text summary of the benchmark report.

    This is the pure-string equivalent of report_results.handle_default(),
    which prints directly to stdout.
    """
    run_id = report.get("run_id", "unknown")
    commit_sha = report.get("commit_sha", "unknown")
    gpus = report.get("gpus", {})

    lines: list[str] = [f"=== Benchmark Report (tag={tag}, run={run_id}, commit={commit_sha}) ===\n"]

    for gpu_key, gpu_data in gpus.items():
        gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
        regressions = sort_regressions(gpu_data.get("regressions", []))
        improvements = sort_improvements(gpu_data.get("improvements", []))
        total_checked = gpu_data.get("total_checked", 0)
        skipped = gpu_data.get("skipped", 0)

        lines.append(f"--- {gpu_name} ({total_checked} checked, {skipped} skipped) ---")
        if regressions:
            lines.append(f"  Regressions: {len(regressions)}")
            for r in regressions[:10]:
                params = format_params(r.get("params", ""))
                lines.append(f"    {r.get('benchmark', '?')} {params}: {r.get('drop_pct', 0):+.1f}%")
            if len(regressions) > 10:
                lines.append(f"    ... and {len(regressions) - 10} more")
        if improvements:
            lines.append(f"  Improvements: {len(improvements)}")
            for r in improvements[:10]:
                params = format_params(r.get("params", ""))
                lines.append(f"    {r.get('benchmark', '?')} {params}: {r.get('drop_pct', 0):+.1f}%")
            if len(improvements) > 10:
                lines.append(f"    ... and {len(improvements) - 10} more")
        if not regressions and not improvements:
            lines.append("  No significant changes.")
        lines.append("")

    return "\n".join(lines)
