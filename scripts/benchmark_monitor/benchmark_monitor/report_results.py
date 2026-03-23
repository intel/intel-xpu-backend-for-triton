"""Read a regression report JSON and post results to GitHub (issues, PR comments).

Supports three modes based on the --tag argument:
  - "ci": Create/update GitHub issues per GPU with regressions.
  - "pr-*": Post/update a PR comment with benchmark results.
  - Other: Print a summary to stdout only.
"""
# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals,unused-argument

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

GPU_SHORT_NAMES: dict[str, str] = {
    "pvc": "PVC",
    "bmg": "BMG",
}

BENCHMARK_MONITOR_MARKER = "<!-- benchmark-monitor -->"
MAX_PR_TABLE_ROWS = 20

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_params(params_json: str) -> str:
    """Parse a JSON params string and return a compact 'K1=V1, K2=V2' representation."""
    try:
        obj = json.loads(params_json)
        return ", ".join(f"{k}={v}" for k, v in obj.items())
    except (json.JSONDecodeError, TypeError):
        return params_json


def _run_gh(args: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a ``gh`` CLI command and return the result.

    Logs warnings on failure but never raises so the rest of the script can continue.
    """
    cmd = ["gh"] + args
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, input=input_text)
        if result.returncode != 0:
            logger.warning("gh command failed (rc=%d): %s\nstderr: %s", result.returncode, " ".join(cmd), result.stderr)
        return result
    except FileNotFoundError:
        logger.error("gh CLI not found. Ensure GitHub CLI is installed and on PATH.")
        return subprocess.CompletedProcess(cmd, returncode=1, stdout="", stderr="gh not found")


def _sort_regressions(items: list[dict]) -> list[dict]:
    """Sort regressions by drop_pct ascending (worst first, most negative first)."""
    return sorted(items, key=lambda r: r.get("drop_pct", 0))


def _sort_improvements(items: list[dict]) -> list[dict]:
    """Sort improvements by gain_pct descending (best improvement first, most positive first)."""
    return sorted(items, key=lambda r: r.get("gain_pct", 0), reverse=True)


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------


def _regression_table_row(item: dict, *, current_label: str = "Current") -> str:
    params = _format_params(item.get("params", ""))
    baseline = item.get("baseline_median", 0)
    current = item.get("current_tflops", 0)
    drop = item.get("drop_pct", 0)
    z_score = item.get("modified_z", 0)
    return f"| {item.get('benchmark', '')} | {params} | {baseline:.1f} | {current:.1f} | {drop:+.1f}% | {z_score:.1f} |"


def _regression_table(items: list[dict], *, current_label: str = "Current", limit: int = 0) -> str:
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


def _improvement_table_row(item: dict, *, current_label: str = "Current") -> str:
    params = _format_params(item.get("params", ""))
    baseline = item.get("baseline_median", 0)
    current = item.get("current_tflops", 0)
    gain = item.get("gain_pct", 0)
    z_score = item.get("modified_z", 0)
    return f"| {item.get('benchmark', '')} | {params} | {baseline:.1f} | {current:.1f} | {gain:+.1f}% | {z_score:.1f} |"


def _improvement_table(items: list[dict], *, current_label: str = "Current", limit: int = 0) -> str:
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
# CI mode: GitHub Issues
# ---------------------------------------------------------------------------


def _driver_change_notice(driver_change: list[dict] | None) -> str:
    if not driver_change:
        return ""
    parts = [f"{c.get('field', '?')}: {c.get('from', '?')} \u2192 {c.get('to', '?')}" for c in driver_change]
    return (f"\n> **Note:** This regression coincides with a driver version change "
            f"({', '.join(parts)}) and may be driver-caused.\n")


def _build_issue_body(
    gpu_key: str,
    gpu_data: dict,
    run_id: str,
    run_url: str,
    commit_sha: str,
    datetime_str: str,
) -> str:
    regressions = _sort_regressions(gpu_data.get("regressions", []))
    improvements = _sort_improvements(gpu_data.get("improvements", []))
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
        _driver_change_notice(driver_change),
        f"### Regressions ({len(regressions)} found, {total_checked} checked, {skipped} skipped)\n",
        _regression_table(regressions),
    ]

    if improvements:
        body_parts.append(f"\n### Improvements ({len(improvements)} found)\n")
        body_parts.append(_improvement_table(improvements))

    return "\n".join(body_parts)


def _find_existing_issue(repo: str, gpu_key: str) -> int | None:
    """Search for an open perf-regression issue for the given GPU. Returns the issue number or None."""
    gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
    result = _run_gh([
        "issue",
        "list",
        "--repo",
        repo,
        "--label",
        "perf-regression",
        "--state",
        "open",
        "--search",
        gpu_name,
        "--json",
        "number,title",
        "--limit",
        "10",
    ])
    if result.returncode != 0:
        return None
    try:
        issues = json.loads(result.stdout)
        for issue in issues:
            title = issue.get("title", "")
            if title.startswith("[Perf Regression") and gpu_name.lower() in title.lower():
                return issue["number"]
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _create_or_update_issue(
    repo: str,
    gpu_key: str,
    gpu_data: dict,
    run_id: str,
    run_url: str,
    commit_sha: str,
    datetime_str: str,
) -> str | None:
    """Create a new issue or comment on an existing one. Returns the issue URL or None on failure."""
    regressions = gpu_data.get("regressions", [])
    if not regressions:
        return None

    driver_change = gpu_data.get("driver_change")
    gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
    n_regressions = len(regressions)

    body = _build_issue_body(gpu_key, gpu_data, run_id, run_url, commit_sha, datetime_str)

    existing_number = _find_existing_issue(repo, gpu_key)
    if existing_number is not None:
        logger.info("Found existing issue #%d for %s, adding comment.", existing_number, gpu_name)
        result = _run_gh([
            "issue",
            "comment",
            str(existing_number),
            "--repo",
            repo,
            "--body",
            body,
        ])
        if result.returncode == 0:
            return f"https://github.com/{repo}/issues/{existing_number}"
        return None

    if driver_change:
        title = f"[Perf Regression - Driver Change] {n_regressions} benchmarks regressed on {gpu_name}"
    else:
        title = f"[Perf Regression] {n_regressions} benchmarks regressed on {gpu_name}"

    result = _run_gh([
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--label",
        "perf-regression",
        "--body",
        body,
    ])
    if result.returncode == 0:
        issue_url = result.stdout.strip()
        logger.info("Created issue: %s", issue_url)
        return issue_url
    return None


def handle_ci(report: dict, run_url: str, repo: str) -> None:
    """Handle the 'ci' tag: create/update GitHub issues and post to Slack."""
    run_id = report.get("run_id", "unknown")
    commit_sha = report.get("commit_sha", "unknown")
    datetime_str = report.get("datetime", "unknown")
    gpus = report.get("gpus", {})

    issue_urls: dict[str, str | None] = {}

    for gpu_key, gpu_data in gpus.items():
        regressions = gpu_data.get("regressions", [])
        if regressions:
            url = _create_or_update_issue(repo, gpu_key, gpu_data, run_id, run_url, commit_sha, datetime_str)
            issue_urls[gpu_key] = url
        else:
            gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
            logger.info("No regressions on %s, skipping issue creation.", gpu_name)


# ---------------------------------------------------------------------------
# PR mode
# ---------------------------------------------------------------------------


def _build_pr_comment(report: dict) -> str:
    """Build the full PR comment body covering all GPUs."""
    gpus = report.get("gpus", {})
    sections: list[str] = [BENCHMARK_MONITOR_MARKER, "## Benchmark Results\n"]

    for gpu_key, gpu_data in gpus.items():
        gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
        total_checked = gpu_data.get("total_checked", 0)
        skipped = gpu_data.get("skipped", 0)
        regressions = _sort_regressions(gpu_data.get("regressions", []))
        improvements = _sort_improvements(gpu_data.get("improvements", []))

        sections.append(f"### {gpu_name} ({total_checked} benchmarks, {skipped} skipped)\n")

        if not regressions and not improvements:
            sections.append(":white_check_mark: No significant performance changes detected\n")
            continue

        if regressions:
            table = _regression_table(regressions, current_label="PR", limit=MAX_PR_TABLE_ROWS)
            sections.append(f"<details><summary>:red_circle: Regressions ({len(regressions)} found)</summary>\n")
            sections.append(table)
            sections.append("\n</details>\n")

        if improvements:
            table = _improvement_table(improvements, current_label="PR", limit=MAX_PR_TABLE_ROWS)
            sections.append(f"<details><summary>:green_circle: Improvements ({len(improvements)} found)</summary>\n")
            sections.append(table)
            sections.append("\n</details>\n")

    return "\n".join(sections)


def _find_existing_pr_comment(repo: str, pr_number: str) -> int | None:
    """Find an existing benchmark-monitor comment on the PR. Returns the comment ID or None."""
    result = _run_gh([
        "api",
        f"repos/{repo}/issues/{pr_number}/comments",
        "--paginate",
        "--jq",
        f'[.[] | select(.body | contains("{BENCHMARK_MONITOR_MARKER}"))][0].id',
    ])
    if result.returncode != 0:
        return None
    comment_id_str = result.stdout.strip()
    if comment_id_str and comment_id_str != "null":
        try:
            return int(comment_id_str)
        except ValueError:
            pass
    return None


def handle_pr(report: dict, pr_number: str, repo: str) -> None:
    """Handle a PR tag: post or update a comment on the PR."""
    body = _build_pr_comment(report)

    existing_comment_id = _find_existing_pr_comment(repo, pr_number)
    if existing_comment_id is not None:
        logger.info("Updating existing PR comment %d.", existing_comment_id)
        _run_gh([
            "api",
            f"repos/{repo}/issues/comments/{existing_comment_id}",
            "--method",
            "PATCH",
            "--field",
            f"body={body}",
        ])
    else:
        logger.info("Creating new PR comment on #%s.", pr_number)
        _run_gh([
            "pr",
            "comment",
            pr_number,
            "--repo",
            repo,
            "--body",
            body,
        ])


# ---------------------------------------------------------------------------
# Default mode: stdout summary
# ---------------------------------------------------------------------------


def handle_default(report: dict, tag: str) -> None:
    """Print a summary to stdout for non-ci, non-pr tags."""
    run_id = report.get("run_id", "unknown")
    commit_sha = report.get("commit_sha", "unknown")
    gpus = report.get("gpus", {})

    print(f"=== Benchmark Report (tag={tag}, run={run_id}, commit={commit_sha}) ===\n")

    for gpu_key, gpu_data in gpus.items():
        gpu_name = GPU_SHORT_NAMES.get(gpu_key, gpu_key.upper())
        regressions = _sort_regressions(gpu_data.get("regressions", []))
        improvements = _sort_improvements(gpu_data.get("improvements", []))
        total_checked = gpu_data.get("total_checked", 0)
        skipped = gpu_data.get("skipped", 0)

        print(f"--- {gpu_name} ({total_checked} checked, {skipped} skipped) ---")
        if regressions:
            print(f"  Regressions: {len(regressions)}")
            for r in regressions[:10]:
                params = _format_params(r.get("params", ""))
                print(f"    {r.get('benchmark', '?')} {params}: {r.get('drop_pct', 0):+.1f}%")
            if len(regressions) > 10:
                print(f"    ... and {len(regressions) - 10} more")
        if improvements:
            print(f"  Improvements: {len(improvements)}")
            for r in improvements[:10]:
                params = _format_params(r.get("params", ""))
                print(f"    {r.get('benchmark', '?')} {params}: {r.get('drop_pct', 0):+.1f}%")
            if len(improvements) > 10:
                print(f"    ... and {len(improvements) - 10} more")
        if not regressions and not improvements:
            print("  No significant changes.")
        print()
