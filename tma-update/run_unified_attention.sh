#!/usr/bin/env bash
#
# run_unified_attention.sh — driver for the unified-attention A/B test.
#
# Same three cases and same output format as run_benchmark.sh, but the kernel
# under test is vLLM's real `unified_attention` instead of the synthetic
# paged_kv_load reproducer:
#
#   run_unified_attention.sh --probe      What does the installed vLLM support?
#                                          Run this FIRST. No GPU.
#   run_unified_attention.sh --ir         Compile-only: count tensormap_create
#                                          for base vs opt. Answers "does the
#                                          optimization apply, and is it
#                                          selective?" without timing anything.
#   run_unified_attention.sh --smoke-test  Launch once, verify vs torch. GPU.
#   run_unified_attention.sh --bench       Time the cases into one table. GPU.
#
#   --case base|tofp|opt|all   single case, or all three (default all)
#
#     base  TMA, optimization OFF (TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1)
#     tofp  no descriptors (use_td=False -> upstream's masked tl.load path)
#     opt   TMA, optimization ON (default Hopper+; the pass demotes
#           loop-recreated descriptors to pointers)
#
#   --td-mode kv|all      which descriptors the kernel creates (default kv)
#     kv   use_td=True             -> in-loop K/V only. The pass demotes all of
#          them, so tensormap_create goes to zero. Proves the pass FIRES.
#     all  use_td=True + use_td_qo -> also out-of-loop Q/O. The pass must KEEP
#          those (hoistable) and demote K/V, so the count drops without
#          reaching zero. Proves the pass is SELECTIVE. Requires the pin to
#          accept use_td_qo — check with --probe.
#
# Common flags:
#   --python <bin>   interpreter with triton AND vllm importable
#                    (default: auto-detect; or set TMA_UPDATE_PYTHON)
#   --warmup N       bench warmup iters (default 50)
#   --trials N       bench timing iters (default 100)
#   --no-verify      skip the correctness gate (not recommended: base/opt/tofp
#                    are three lowerings of the same math, so a demotion bug
#                    reads as a speedup)
#
# Notes:
#   * vLLM must be installed — the kernel is imported, never vendored.
#     scripts/vllm/install-vllm.sh hardcodes VLLM_TARGET_DEVICE=xpu and will NOT
#     work here; see RUN.md ('NVIDIA vLLM install') for the recipe.
#   * TMA requires compute capability >= 9.0. On sm<90 every descriptor is
#     demoted in make_ttir, so all three cases collapse to the same code.
#   * Needs the loop-recreated-only pass built into libtriton — the change this
#     benchmark exists to validate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/unified_attention_nv.py"

MODE=""
PYTHON_BIN="${TMA_UPDATE_PYTHON:-}"
BENCH_CASE="all"
TD_MODE="kv"
WARMUP=50
TRIALS=100
VERIFY_ARGS=()

die() { echo "[run_unified_attention] error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --probe)      MODE="probe"; shift ;;
    --ir)         MODE="ir";    shift ;;
    --smoke-test) MODE="smoke"; shift ;;
    --bench)      MODE="bench"; shift ;;
    --case)       BENCH_CASE="${2:?}"; shift 2 ;;
    --td-mode)    TD_MODE="${2:?}";    shift 2 ;;
    --python)     PYTHON_BIN="${2:?}"; shift 2 ;;
    --warmup)     WARMUP="${2:?}";     shift 2 ;;
    --trials)     TRIALS="${2:?}";     shift 2 ;;
    --no-verify)  VERIFY_ARGS+=(--no-verify); shift ;;
    -h|--help)    sed -n '2,50p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

[[ -n "${MODE}" ]] || die "pick one of --probe | --ir | --smoke-test | --bench"
case "${BENCH_CASE}" in all|base|tofp|opt) ;; *) die "--case must be one of: all base tofp opt (got '${BENCH_CASE}')" ;; esac
case "${TD_MODE}"    in kv|all)           ;; *) die "--td-mode must be kv or all (got '${TD_MODE}')" ;; esac

# --- locate a python with both triton and vllm importable --------------------
# Both, not just triton: the whole point is that we import the kernel rather
# than vendoring it, so a triton-only interpreter fails later and less clearly.
if [[ -z "${PYTHON_BIN}" ]]; then
  for cand in "${CONDA_PREFIX:-}/bin/python" python3 python; do
    [[ -z "${cand}" ]] && continue
    if "${cand}" -c "import triton, vllm" >/dev/null 2>&1; then PYTHON_BIN="${cand}"; break; fi
  done
fi
[[ -n "${PYTHON_BIN}" ]] || die "no python with both 'triton' and 'vllm' importable;
  pass --python or set TMA_UPDATE_PYTHON. If triton imports but vllm does not,
  see RUN.md ('NVIDIA vLLM install') — install-vllm.sh is XPU-only."

echo "[run_unified_attention] python=${PYTHON_BIN} mode=${MODE} td-mode=${TD_MODE}" >&2

# --- preflight: the env failures we actually hit, with fixes -----------------
# --probe is deliberately exempt: diagnosing a broken env is its whole job, so
# gating it behind the same checks would hide the report we want to read.
if [[ "${MODE}" != "probe" ]]; then
  "${PYTHON_BIN}" - <<'PY' || die "preflight failed — see above"
import os, sys

def fail(msg):
    print(f"[preflight] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

try:
    import triton
except Exception as e:
    fail(f"cannot import triton ({e})")

# No set_allocator => torch's bundled 3.2.0 wheel shadowed the source build.
if not hasattr(triton, "set_allocator"):
    fail(f"triton {getattr(triton,'__version__','?')} at {triton.__file__} lacks "
         "set_allocator — this is torch's bundled wheel, not a TMA-capable "
         "build. Fix: `pip install -e .` from your triton checkout (torch "
         "FIRST, editable triton LAST).")

try:
    import torch
except Exception as e:
    fail(f"cannot import torch ({e})")

if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False — no NVIDIA GPU visible.")

cc = torch.cuda.get_device_capability()
print(f"[preflight] OK: triton {triton.__version__} @ "
      f"{os.path.realpath(triton.__file__)}", file=sys.stderr)
print(f"[preflight] OK: torch {torch.__version__}, GPU sm_{cc[0]}{cc[1]} "
      f"({torch.cuda.get_device_name(0)})", file=sys.stderr)
if cc[0] < 9:
    fail(f"sm_{cc[0]}{cc[1]} has no TMA. Below sm_90 the pipeline demotes every "
         "descriptor in make_ttir, so base/opt/tofp compile to the same code "
         "and the comparison is meaningless. Needs Hopper or Blackwell.")
PY
fi

cd "${REPO_ROOT}"

case "${MODE}" in
  probe)
    exec "${PYTHON_BIN}" "${PY_SCRIPT}" probe
    ;;
  ir)
    exec "${PYTHON_BIN}" "${PY_SCRIPT}" ir --td-mode "${TD_MODE}"
    ;;
  smoke)
    CASE="${BENCH_CASE}"; [[ "${CASE}" == "all" ]] && CASE="opt"
    exec "${PYTHON_BIN}" "${PY_SCRIPT}" smoke --case "${CASE}" --td-mode "${TD_MODE}"
    ;;
  bench)
    # Compile-side check first: if the pass did not change the IR, the timings
    # are three measurements of the same code and the table would be noise
    # dressed up as a result.
    echo "[run_unified_attention] IR gate: confirming the pass changes the IR" >&2
    if ! "${PYTHON_BIN}" "${PY_SCRIPT}" ir --td-mode "${TD_MODE}"; then
      die "IR gate failed — the optimization did not toggle. Timings would be
  meaningless, so stopping here. See the [ir] output above."
    fi
    exec "${PYTHON_BIN}" "${PY_SCRIPT}" bench --case "${BENCH_CASE}" \
        --td-mode "${TD_MODE}" --warmup "${WARMUP}" --trials "${TRIALS}" \
        "${VERIFY_ARGS[@]+"${VERIFY_ARGS[@]}"}"
    ;;
esac
