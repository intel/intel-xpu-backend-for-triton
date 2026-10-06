#!/usr/bin/env bash

set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly VLLM_PROJ="$ROOT/vllm"

readonly USAGE="Usage: $(basename "$0") <test_dir|test_file> <out_dir> [pytest args ...]"
readonly TARGET="${1:?$USAGE}"
readonly OUT_DIR="${2:?$USAGE}"
shift 2

readonly RESULTS="$OUT_DIR/results.txt"

if [[ ! -d "$VLLM_PROJ" ]]; then
  echo "Error: vllm project not found: $VLLM_PROJ" >&2
  exit 1
fi

if [[ ! -e "$TARGET" ]]; then
  echo "Error: target not found: $TARGET" >&2
  exit 1
fi

echo "target:  $TARGET"
echo "out_dir: $OUT_DIR"

if [[ -d "$OUT_DIR" && -n "$(ls -A "$OUT_DIR")" ]]; then
  echo "Error: $OUT_DIR is not empty: $(ls -A "$OUT_DIR")" >&2
  exit 1
fi

REL_TARGET="$(realpath --relative-to="$VLLM_PROJ" "$TARGET")"
if [[ "$REL_TARGET" == ..* ]]; then
  echo "Error: target is not inside $VLLM_PROJ: $TARGET" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

mapfile -t TEST_FILES < <(cd "$VLLM_PROJ" && find "$REL_TARGET" -name 'test_*.py' | sort)

for test_file in "${TEST_FILES[@]}"; do
  out_file="$OUT_DIR/$test_file"
  mkdir -p "$(dirname "$out_file")"
  echo "$test_file"
  rc=0
  VLLM_USE_V2_MODEL_RUNNER=1 VLLM_KERNEL_TEST_MAP_OUT="${out_file}.json" \
    pytest "$VLLM_PROJ/$test_file" --continue-on-collection-errors --verbose --tb=no --timeout=600 \
    "$@" > "${out_file}.log" 2>&1 || rc=$?
  echo "$rc $test_file" >> "$RESULTS"
done
