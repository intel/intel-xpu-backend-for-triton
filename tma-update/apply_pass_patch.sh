#!/usr/bin/env bash
#
# apply_pass_patch.sh — apply the loop-recreated-descriptor TMA fallback pass to
# an upstream triton-lang/triton checkout.
#
# This is a STATIC snapshot patch (tma-update/tma-loop-td-pass.patch) covering
# the six compiler files that implement the optimization:
#   include/triton/Dialect/Triton/Transforms/Passes.td            (new pass def)
#   lib/Dialect/Triton/Transforms/RewriteLoopTensorDescriptors.cpp (NEW file)
#   lib/Dialect/Triton/Transforms/CMakeLists.txt                  (register .cpp)
#   python/src/passes.cc                                          (py binding)
#   python/triton/knobs.py                                        (2 knobs)
#   third_party/nvidia/backend/compiler.py                        (make_ttir gate)
#
# It does NOT include the tma-update/ benchmark harness (that stays here).
#
# Usage:
#   apply_pass_patch.sh --triton-root /path/to/triton [--check] [--reverse]
#
#   --check     dry-run only (git apply --check); report and exit, no changes
#   --reverse   un-apply the patch (git apply -R)
#
# After applying you MUST rebuild libtriton (the .cpp / .td / CMake / binding are
# C++): `pip install -e .` from the triton root. The knobs.py + compiler.py
# edits are pure-Python and take effect without a rebuild, but the pass they call
# won't exist until the C++ is built.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PATCH="${SCRIPT_DIR}/tma-loop-td-pass.patch"

TRITON_ROOT=""
CHECK=0
REVERSE=0

die() { echo "[apply_pass_patch] error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --triton-root) TRITON_ROOT="${2:?}"; shift 2 ;;
    --check)       CHECK=1; shift ;;
    --reverse)     REVERSE=1; shift ;;
    -h|--help)     sed -n '2,28p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

[[ -n "${TRITON_ROOT}" ]] || die "pass --triton-root /path/to/triton"
[[ -d "${TRITON_ROOT}/.git" ]] || die "${TRITON_ROOT} is not a git repo root"
[[ -f "${PATCH}" ]] || die "patch not found: ${PATCH}"

# The patch is rooted at the triton repo top (a/include/..., a/lib/..., etc.),
# so apply with -p1 from the repo root.
GIT_APPLY=(git -C "${TRITON_ROOT}" apply -p1)
DIR=()
if (( REVERSE )); then DIR+=(--reverse); fi

echo "[apply_pass_patch] target : ${TRITON_ROOT}" >&2
echo "[apply_pass_patch] patch  : ${PATCH}" >&2
echo "[apply_pass_patch] mode   : $( ((REVERSE)) && echo reverse || echo forward )$( ((CHECK)) && echo ' (check only)')" >&2

# Always dry-run first so a failure reports cleanly instead of half-applying.
if ! "${GIT_APPLY[@]}" "${DIR[@]}" --check "${PATCH}" 2>/tmp/apply_pass.err; then
  echo "[apply_pass_patch] DRY-RUN FAILED — patch does not apply cleanly:" >&2
  sed 's/^/    /' /tmp/apply_pass.err >&2
  echo "  The upstream files likely drifted from the snapshot. Re-generate the" >&2
  echo "  patch, or apply the six edits by hand (see the patch header)." >&2
  exit 1
fi
echo "[apply_pass_patch] dry-run OK — patch applies cleanly." >&2

if (( CHECK )); then
  echo "[apply_pass_patch] --check: not modifying anything." >&2
  exit 0
fi

"${GIT_APPLY[@]}" "${DIR[@]}" "${PATCH}"
if (( REVERSE )); then
  echo "[apply_pass_patch] reverted." >&2
else
  echo "[apply_pass_patch] applied. NEXT: rebuild libtriton —" >&2
  echo "    (cd ${TRITON_ROOT} && pip install -e .)" >&2
fi
