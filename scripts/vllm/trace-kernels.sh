#!/usr/bin/env bash
#
# Trace which vLLM tests launch which Triton kernels.
#
# Run after `install-vllm.sh`. This script:
#   1. Applies kernel-trace.patch to vllm/tests/conftest.py (idempotent).
#   2. Sets VLLM_TRITON_KERNEL_TRACE so the conftest hook records every
#      Triton kernel launch -> {set of pytest nodeids that triggered it}.
#   3. Runs pytest on the requested test paths.
#   4. Prints a kernel-by-kernel summary of the resulting JSON.
#
# Usage:
#   scripts/vllm/trace-kernels.sh [-o OUT_JSON] [-k KERNEL_NAMES] -- <pytest args>
#
# Examples:
#   scripts/vllm/trace-kernels.sh -- tests/kernels/moe/test_batched_moe.py
#   scripts/vllm/trace-kernels.sh -k fused_moe_kernel,unified_attention_kernel \
#       -- tests/kernels/
#   scripts/vllm/trace-kernels.sh -o /tmp/run1.json -- tests/kernels -x

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
readonly VLLM_PROJ="$ROOT/vllm"
readonly PATCH_FILE="$SCRIPT_DIR/kernel-trace.patch"

out_json=""
kernel_filter=""
pytest_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output)
      out_json="$2"
      shift 2
      ;;
    -k|--kernels)
      kernel_filter="$2"
      shift 2
      ;;
    --)
      shift
      pytest_args=("$@")
      break
      ;;
    -h|--help)
      sed -n '3,20p' "$0"
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1 (did you forget '--' before pytest args?)" >&2
      exit 1
      ;;
  esac
done

if [[ ${#pytest_args[@]} -eq 0 ]]; then
  echo "ERROR: no pytest args given. Pass them after '--'." >&2
  echo "  e.g. $0 -- tests/kernels/moe/test_batched_moe.py" >&2
  exit 1
fi

if [[ ! -d "$VLLM_PROJ" ]]; then
  echo "ERROR: $VLLM_PROJ not found. Run scripts/vllm/install-vllm.sh first." >&2
  exit 1
fi

if [[ -z "$out_json" ]]; then
  out_json="$(mktemp -t vllm-kernel-trace.XXXXXX.json)"
fi

# Apply the conftest patch idempotently. `git apply --check` succeeds only if
# the patch has not yet been applied; otherwise we assume it is already in.
if git -C "$VLLM_PROJ" apply --check "$PATCH_FILE" 2>/dev/null; then
  echo "*** Applying $PATCH_FILE to $VLLM_PROJ. ***"
  git -C "$VLLM_PROJ" apply "$PATCH_FILE"
else
  echo "*** kernel-trace.patch already applied (or conflicts); continuing. ***"
fi

cd "$VLLM_PROJ"

env_args=(VLLM_TRITON_KERNEL_TRACE="$out_json")
if [[ -n "$kernel_filter" ]]; then
  env_args+=(VLLM_TRITON_KERNEL_TRACE_NAMES="$kernel_filter")
fi

echo "*** Trace output: $out_json ***"
echo "*** Running: ${env_args[*]} python -m pytest ${pytest_args[*]} ***"
set +e
env "${env_args[@]}" python -m pytest "${pytest_args[@]}"
pytest_status=$?
set -e

if [[ ! -s "$out_json" ]]; then
  echo "*** No trace data written to $out_json. ***"
  echo "*** (Did any test actually launch a Triton kernel?) ***"
  exit "$pytest_status"
fi

echo
echo "============================================================"
echo " Kernel -> tests summary  ($out_json)"
echo "============================================================"
python - "$out_json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)

if not data:
    print("(empty: no kernel launches recorded)")
    sys.exit(0)

name_w = max(len(k) for k in data)
total_launches = sum(len(v) for v in data.values())
print(f"{len(data)} unique kernel(s), {total_launches} (kernel, test) pair(s)")
print()
print(f"{'KERNEL'.ljust(name_w)}  TESTS")
print(f"{'-' * name_w}  -----")
for kernel, tests in sorted(data.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    print(f"{kernel.ljust(name_w)}  {len(tests)}")

print()
print("------------------------------------------------------------")
print(" Detail")
print("------------------------------------------------------------")
for kernel, tests in sorted(data.items()):
    print(f"\n{kernel}  ({len(tests)} test(s))")
    for t in tests:
        print(f"  - {t}")
PY

exit "$pytest_status"
