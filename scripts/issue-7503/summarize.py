"""
Tabulates the records `test_chain_reduce_ptss.py` writes to `ISSUE_7503_RESULTS_DIR`.

Prints a Markdown table, one row per case that overflowed, failed or spilled in any GRF mode or
compile path, and appends it to the GitHub Actions job summary when `GITHUB_STEP_SUMMARY` is set.
"""

import argparse
import json
import os
from pathlib import Path

COLUMNS = [(path, grf_mode) for path in ("jit", "aot") for grf_mode in ("default", "128", "256")]
ENV_FIELDS = ("device", "driver_version", "arch", "torch", "triton", "platform", "ocloc")


def case_key(record: dict) -> str:
    # The test_core.py::test_chain_reduce id of the case.
    first_axis, op, layout, m, n = (record[key] for key in ("first_axis", "op", "layout", "M", "N"))
    return f"{first_axis}-{op}-src_layout{layout}-{m}-{n}"


def sort_key(record: dict) -> tuple:
    return tuple(record[key] for key in ("layout", "first_axis", "op", "M", "N"))


def format_cell(record: dict | None) -> str:
    if record is None:
        return "-"
    status = record["status"]
    if status == "ok":
        n_regs, n_spills = record["n_regs"], record["n_spills"]
        spill = f", spill {n_spills} dw/lane" if n_spills > 0 else ", spill unknown" if n_spills < 0 else ""
        return f"ok, {n_regs} GRF{spill}"
    if status == "ptss":
        required = " / ".join(str(req) for req, _ in record["ptss"])
        return f"PTSS {required} B" if required else "PTSS (no numbers)"
    if status == "wrong":
        actual, expected = record["actual"], record["expected"]
        return f"WRONG {actual} != {expected}"
    return "ERROR"


def is_interesting(record: dict) -> bool:
    return record["status"] != "ok" or record["n_spills"] != 0


def render_environment(env: dict) -> list[str]:
    if not env:
        return []
    device, driver, arch, torch_version, triton_version, system, ocloc = (env.get(key, "?") for key in ENV_FIELDS)
    ocloc_text = f"`{ocloc}`" if ocloc else "not on PATH, AOT path skipped"
    return [
        f"{device}, driver {driver}, arch `{arch}`, torch {torch_version}, triton {triton_version}, {system}; "
        f"ocloc: {ocloc_text}", ""
    ]


def render_counts(records: list[dict]) -> list[str]:
    counts: dict[str, int] = {}
    for record in records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    outcomes = ", ".join(f"{status} {n}" for status, n in sorted(counts.items()))
    limits = ", ".join(sorted({str(limit) for r in records if r["status"] == "ptss" for _, limit in r["ptss"]}))
    limit_text = f". PTSS limit reported by IGC: {limits} B." if limits else "."
    return [f"Compiles by outcome: {outcomes}{limit_text}", ""]


def render_table(records: list[dict]) -> list[str]:
    cases: dict[str, dict] = {}
    for record in sorted(records, key=sort_key):
        cases.setdefault(case_key(record), {})[(record["path"], record["grf_mode"])] = record
    interesting = {key: cells for key, cells in cases.items() if any(is_interesting(r) for r in cells.values())}

    header = " | ".join(f"{path} {grf_mode}" for path, grf_mode in COLUMNS)
    lines = [
        "Cases that overflowed, failed or spilled in any mode (`PTSS <n> B` is the scratch IGC asked for; "
        "`dw/lane` is `n_spills`):", "", f"| case | {header} |", "|---" * (len(COLUMNS) + 1) + "|"
    ]
    for key, cells in interesting.items():
        row = " | ".join(format_cell(cells.get(column)) for column in COLUMNS)
        lines.append(f"| `{key}` | {row} |")
    quiet = len(cases) - len(interesting)
    lines += ["", f"Cases not listed, which compiled and ran correctly in every mode without spilling: {quiet}.", ""]
    return lines


def render_errors(records: list[dict]) -> list[str]:
    errors = [r for r in records if r["status"] == "error"]
    if not errors:
        return []
    lines = ["<details><summary>Errors</summary>", ""]
    for record in errors:
        case_id = record["id"]
        first_line = (record["message"].splitlines() or [""])[0][:300]
        lines.append(f"- `{case_id}`: {first_line}")
    return lines + ["", "</details>", ""]


def render(results_dir: Path, title: str) -> str:
    env_file = results_dir / "env.json"
    env = json.loads(env_file.read_text()) if env_file.exists() else {}
    records = [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json")) if f.name != "env.json"]

    lines = [f"### {title}", ""] + render_environment(env)
    if not records:
        lines.append("No records found: the sweep did not run, or `ISSUE_7503_RESULTS_DIR` was not set.")
    else:
        lines += render_counts(records) + render_table(records) + render_errors(records)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate the issue #7503 sweep results.")
    parser.add_argument("results_dir", type=Path, help="directory the sweep wrote its JSON records to")
    parser.add_argument("--title", default="Issue 7503 sweep", help="heading of the table")
    args = parser.parse_args()

    summary = render(args.results_dir, args.title)
    print(summary)
    if step_summary := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(step_summary, "a", encoding="utf-8") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
