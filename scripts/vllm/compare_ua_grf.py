"""Benchmark attention decode pruning with one compiler and GPU job."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

ARMS = ("control", "decode_pruned")
PATCH = "benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch"
KEYS = ("vllm-unified-attention", "vllm-unified-attention-fp8")


def git_output(repo, *args):
    """Read Git output from an explicit repository."""
    return subprocess.check_output(["git", *args], cwd=repo)


def validate_refs(repo, refs):
    """Require exact commits whose only difference is the attention patch."""
    if set(refs) != set(ARMS):
        raise ValueError(f"Expected exactly these arms: {ARMS}")
    for ref in refs.values():
        if not isinstance(ref, str) or not re.fullmatch(r"[0-9a-f]{40}", ref):
            raise ValueError("Comparison refs must be full commit hashes")
        if git_output(repo, "rev-parse", f"{ref}^{{commit}}").decode().strip() != ref:
            raise ValueError("Comparison ref is not a commit")
    if git_output(repo, "rev-parse", "HEAD").decode().strip() != refs["control"]:
        raise ValueError("Run the comparison from the control commit")
    patches = {}
    for arm in ARMS:
        changed = git_output(repo, "diff", "--name-only", refs["control"], refs[arm]).decode().splitlines()
        if changed != ([] if arm == "control" else [PATCH]):
            raise ValueError(f"Unexpected source differences for {arm}: {changed}")
        patches[arm] = git_output(repo, "show", f"{refs[arm]}:{PATCH}")
    return patches


def save_manifest(path, manifest):
    """Write progress atomically so partial results remain readable."""
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(manifest, indent=2) + "\n")
    pending.replace(path)


def run_comparison(repo, refs, patches, reports, tag):
    """Apply each arm to the same vLLM checkout and collect separate reports."""
    vllm = repo / "vllm"
    pin = (repo / "scripts/vllm/vllm-pin.txt").read_text().strip()
    if git_output(vllm, "rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError("Installed vLLM checkout does not match the comparison pin")
    reports.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "refs": refs, "compiler_commit": refs["control"], "vllm_pin": pin, "runner":
        os.environ.get("RUNNER_NAME"), "run_id": os.environ.get("GITHUB_RUN_ID"), "arms": {}
    }
    manifest_path = reports / "grf-comparison.json"
    save_manifest(manifest_path, manifest)
    with tempfile.TemporaryDirectory(prefix="ua-grf-", dir=os.environ.get("RUNNER_TEMP")) as temporary:
        temporary = Path(temporary)
        try:
            for arm in ARMS:
                (reports / arm).mkdir()
                patch = temporary / f"{arm}.patch"
                patch.write_bytes(patches[arm])
                item = {
                    "status": "running", "commit": refs[arm], "started": time.time(), "patch_sha256":
                    hashlib.sha256(patches[arm]).hexdigest()
                }
                manifest["arms"][arm] = item
                save_manifest(manifest_path, manifest)
                subprocess.run(["git", "apply", "--check", str(patch)], cwd=vllm, check=True)
                subprocess.run(["git", "apply", str(patch)], cwd=vllm, check=True)
                try:
                    for key in KEYS:
                        print(f"GRF comparison: {arm}, {key}, commit {refs[arm]}", flush=True)
                        subprocess.run(
                            ["triton-benchmarks", "run", key, "--reports",
                             str(reports / arm), "--tag", f"{tag}-{arm}"], cwd=vllm, env=dict(
                                 os.environ, TD_PATCHED="1", FP8="1" if key.endswith("-fp8") else "0",
                                 PYTHONPATH=os.pathsep.join(filter(None, (str(vllm), os.environ.get("PYTHONPATH")))),
                                 TRITON_CACHE_DIR=str(temporary / arm / "cache")), check=True)
                finally:
                    subprocess.run(["git", "apply", "--reverse", str(patch)], cwd=vllm, check=True)
                item.update(status="complete", finished=time.time())
                save_manifest(manifest_path, manifest)
        except Exception as error:
            manifest.update(status="failed", error=str(error))
            save_manifest(manifest_path, manifest)
            raise
    manifest.update(status="complete", finished=time.time())
    save_manifest(manifest_path, manifest)


def main():
    """Validate the supplied commit map and optionally run GPU benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refs", required=True, help="JSON map from arm names to exact commit hashes")
    parser.add_argument("--reports", type=Path)
    parser.add_argument("--tag", default="ua-grf-comparison")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    refs = json.loads(args.refs)
    patches = validate_refs(repo, refs)
    if args.validate_only:
        print(json.dumps({arm: hashlib.sha256(patch).hexdigest() for arm, patch in patches.items()}, indent=2))
        return
    if args.reports is None:
        parser.error("--reports is required unless --validate-only is used")
    run_comparison(repo, refs, patches, args.reports.resolve(), args.tag)


if __name__ == "__main__":
    main()
