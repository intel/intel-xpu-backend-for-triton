"""Bootstrap historical benchmark data from GitHub Actions artifacts.

Downloads benchmark report artifacts from recent workflow runs and populates
the history.json files used by the regression detection system. This is a
one-time setup script to seed the benchmark-data branch with historical data.

Usage:
    python scripts/benchmark-monitor/bootstrap_history.py \
        --history-dir benchmark-data \
        --max-runs 50 \
        --workflow triton-benchmarks.yml
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "intel/intel-xpu-backend-for-triton"
WORKFLOW = "triton-benchmarks.yml"


def run_gh(args: list[str], capture: bool = True) -> subprocess.CompletedProcess:
    """Run a gh CLI command and return the result."""
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=capture, text=True, check=False)
    if result.returncode != 0:
        print(f"  gh command failed: {' '.join(cmd)}", file=sys.stderr)
        if result.stderr:
            print(f"  stderr: {result.stderr.strip()}", file=sys.stderr)
    return result


def list_successful_runs(workflow: str, max_runs: int, actor: str) -> list[dict]:
    """List successful workflow runs from GitHub Actions."""
    cmd = [
        "run",
        "list",
        "--repo",
        REPO,
        "--workflow",
        workflow,
        "--status",
        "success",
        "--limit",
        str(max_runs),
        "--json",
        "databaseId,createdAt,headSha,displayTitle",
    ]
    if actor:
        cmd.extend(["--user", actor])

    result = run_gh(cmd)
    if result.returncode != 0:
        return []
    return json.loads(result.stdout)


def download_artifacts(run_id: int, dest_dir: Path) -> bool:
    """Download benchmark-reports artifact for a given run."""
    result = run_gh([
        "run",
        "download",
        str(run_id),
        "--repo",
        REPO,
        "--name",
        "benchmark-reports",
        "--dir",
        str(dest_dir),
    ])
    return result.returncode == 0


def detect_gpu(reports_dir: Path) -> str | None:
    """Detect GPU type from report CSV files by reading gpu_device column."""
    import csv

    for csv_file in reports_dir.glob("*-report.csv"):
        try:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    device = row.get("gpu_device", "")
                    if "Max 1550" in device or "Max 1100" in device:
                        return "pvc"
                    if "B580" in device or "BMG" in device:
                        return "bmg"
                    break  # Only need first row
        except (OSError, csv.Error):
            continue
    return None


def parse_reports(reports_dir: Path, run_id: int, run_info: dict) -> dict | None:
    """Parse report CSVs into a history entry. Returns None if no triton reports found."""
    import csv

    results: dict[str, dict] = {}
    metadata: dict[str, str] = {}

    for csv_file in sorted(reports_dir.glob("*-report.csv")):
        try:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not metadata:
                        metadata = {
                            "agama_version": row.get("agama_version", ""),
                            "libigc1_version": row.get("libigc1_version", ""),
                            "datetime": row.get("datetime", run_info.get("createdAt", "")),
                        }

                    benchmark = row.get("benchmark", "")
                    compiler = row.get("compiler", "")
                    params = row.get("params", "{}")
                    tflops_str = row.get("tflops", "")

                    if compiler != "triton" or not tflops_str:
                        continue

                    try:
                        tflops = float(tflops_str)
                    except ValueError:
                        continue

                    key = f"{benchmark}/{compiler}/{params}"
                    entry: dict[str, float] = {"tflops": tflops}

                    # TODO: Collect and analyze hbm_gbs (memory bandwidth) metric once
                    # detect_regressions.py supports multi-metric analysis.

                    results[key] = entry
        except (OSError, csv.Error) as e:
            print(f"  Warning: failed to parse {csv_file}: {e}", file=sys.stderr)
            continue

    if not results:
        return None

    return {
        "run_id": str(run_id),
        "datetime": metadata.get("datetime", ""),
        "tag": "ci",
        "commit_sha": run_info.get("headSha", ""),
        "agama_version": metadata.get("agama_version", ""),
        "libigc1_version": metadata.get("libigc1_version", ""),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap benchmark history from GitHub Actions artifacts")
    parser.add_argument("--history-dir", required=True, help="Path to benchmark-data directory")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs to download")
    parser.add_argument("--workflow", default=WORKFLOW, help="Workflow filename")
    parser.add_argument("--actor", default="glados-intel", help="Filter by actor (default: glados-intel)")
    parser.add_argument("--dry-run", action="store_true", help="List runs without downloading")
    args = parser.parse_args()

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing up to {args.max_runs} successful runs for {args.workflow}...")
    runs = list_successful_runs(args.workflow, args.max_runs, args.actor)
    if not runs:
        print("No runs found.")
        return

    print(f"Found {len(runs)} runs.")

    if args.dry_run:
        for run in runs:
            print(f"  Run {run['databaseId']} | {run['createdAt']} | {run['headSha'][:8]} | {run['displayTitle']}")
        return

    # Load existing histories
    histories: dict[str, list] = {}
    for gpu in ("pvc", "bmg"):
        history_file = history_dir / gpu / "history.json"
        if history_file.exists():
            histories[gpu] = json.loads(history_file.read_text())
        else:
            histories[gpu] = []

    existing_run_ids = set()
    for gpu_history in histories.values():
        for entry in gpu_history:
            existing_run_ids.add(str(entry.get("run_id", "")))

    imported = 0
    skipped = 0

    for run in reversed(runs):  # Process oldest first for chronological order
        run_id = run["databaseId"]

        if str(run_id) in existing_run_ids:
            print(f"  Run {run_id}: already in history, skipping.")
            skipped += 1
            continue

        print(f"  Run {run_id} ({run['createdAt'][:10]}): downloading...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "reports"
            if not download_artifacts(run_id, dest):
                print("download failed, skipping.")
                continue

            gpu = detect_gpu(dest)
            if not gpu:
                print("could not detect GPU, skipping.")
                continue

            entry = parse_reports(dest, run_id, run)
            if not entry:
                print("no triton results found, skipping.")
                continue

            histories[gpu].append(entry)
            existing_run_ids.add(str(run_id))
            imported += 1
            n_results = len(entry["results"])
            print(f"imported {n_results} metrics for {gpu.upper()}.")

    # Write updated histories
    for gpu, history in histories.items():
        gpu_dir = history_dir / gpu
        gpu_dir.mkdir(parents=True, exist_ok=True)
        history_file = gpu_dir / "history.json"
        history_file.write_text(json.dumps(history, indent=2) + "\n")
        print(f"Wrote {len(history)} entries to {history_file}")

    print(f"\nDone. Imported {imported} runs, skipped {skipped} (already present).")


if __name__ == "__main__":
    main()
