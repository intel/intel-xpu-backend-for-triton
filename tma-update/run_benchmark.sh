#!/usr/bin/env bash
#
# run_benchmark.sh — driver for the TMA descriptor-update A/B test.
#
# One script for the whole flow on an sm_90 (Hopper) or sm_100 (Blackwell) box:
#
#   run_benchmark.sh --generate           Compile the kernel to PTX (baseline).
#                                          No GPU required.
#   run_benchmark.sh --smoke-test         Launch the kernel once and verify its
#                                          output against a torch reference.
#                                          Requires a GPU.
#   run_benchmark.sh --bench              3-case benchmark, each behind a
#                                          correctness gate, into one table:
#                                            base  #1 TMA, optimization OFF
#                                                     (raw tensormap_create/iter)
#                                            tofp  #2 manual tensor-of-pointers
#                                            opt   #3 TMA, optimization ON
#                                                     (loop-recreated-descriptor
#                                                      gate pass demotes to ptrs)
#                                          Requires a GPU.
#
#   run_benchmark.sh --bench --case base|tofp|opt
#                                          Run a SINGLE case only (no table).
#                                          --case all (default) runs all three.
#
#   run_benchmark.sh --calibrate [--case base|tofp|opt] [--burst N]
#                                          Time a long back-to-back burst, report
#                                          run-to-run variance, and recommend
#                                          --warmup / --trials for stable results.
#                                          Requires a GPU. (default case: base)
#
#   run_benchmark.sh --patch [--triton-root DIR] [--patch-check] [--patch-reverse]
#                                          Apply the compiler-side optimization
#                                          patch (tma-loop-td-pass.patch) to an
#                                          upstream triton checkout via
#                                          apply_pass_patch.sh. No GPU needed.
#                                          Default root = tma-update/'s parent
#                                          (i.e. the repo root when this folder
#                                          is copied into upstream triton).
#                                          --patch-check = dry-run only;
#                                          --patch-reverse = un-apply.
#                                          REBUILD after: `pip install -e .`.
#
# The optimization is the `triton-rewrite-loop-recreated-descriptors` compiler
# pass (default-on for Hopper+). base vs opt is the SAME kernel with the pass
# toggled: base sets TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1, opt uses the
# default. No PTX override is involved anymore.
#
# Workload scaling (lift the timing signal above launch/timer overhead):
#   --num-kv-blocks N   loop trip count — scales per-iteration descriptor cost
#                       (the thing measured); this is the knob to grow.
#   --num-seqs N        grid size / CTAs — fills the GPU, amortizes launch cost.
#
# Common flags:
#   --arch sm90|sm100     target arch for --generate (default sm100)
#   --triton-root DIR     --patch target (default: tma-update/'s parent)
#   --python <bin>        python interpreter with triton importable
#                         (default: auto-detect; or set TMA_UPDATE_PYTHON)
#   --warmup N            bench warmup iters (default 50)
#   --trials N            bench timing iters (default 100)
#   --burst N             calibrate burst length (default 500)
#
# Notes:
#   * TMA requires compute capability >= 9.0. Will NOT run on A100/A10G (sm_80/86)
#     or L4/L40S (sm_89) — ptxas rejects the tensormap.* instructions.
#   * Requires the `triton-rewrite-loop-recreated-descriptors` pass to be built
#     into libtriton (the compiler change this benchmark validates).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/paged_kv_load.py"
KERNEL_NAME="paged_kv_load_kernel"

MODE=""
ARCH="sm100"
PYTHON_BIN="${TMA_UPDATE_PYTHON:-}"
WARMUP=50
TRIALS=100
BURST=500
NUM_KV_BLOCKS=""   # empty = kernel default (512)
NUM_SEQS=""        # empty = kernel default (8)
OUT_DIR="${SCRIPT_DIR}/ptx-opt-test"
BENCH_CASE="all"   # all | base | tofp | opt
TRITON_ROOT=""     # --patch target; default = REPO_ROOT (tma-update's parent)
PATCH_ARGS=()      # extra flags forwarded to apply_pass_patch.sh

die() { echo "[run_benchmark] error: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate)    MODE="generate"; shift ;;
    --smoke-test)  MODE="smoke";    shift ;;
    --bench)       MODE="bench";    shift ;;
    --calibrate)   MODE="calibrate"; shift ;;
    --patch)       MODE="patch";    shift ;;
    --triton-root) TRITON_ROOT="${2:?}"; shift 2 ;;
    --patch-check) PATCH_ARGS+=(--check); shift ;;
    --patch-reverse) PATCH_ARGS+=(--reverse); shift ;;
    --case)        BENCH_CASE="${2:?}"; shift 2 ;;
    --burst)       BURST="${2:?}"; shift 2 ;;
    --num-kv-blocks) NUM_KV_BLOCKS="${2:?}"; shift 2 ;;
    --num-seqs)      NUM_SEQS="${2:?}"; shift 2 ;;
    --arch)        ARCH="${2:?}";   shift 2 ;;
    --python)      PYTHON_BIN="${2:?}";    shift 2 ;;
    --warmup)      WARMUP="${2:?}"; shift 2 ;;
    --trials)      TRIALS="${2:?}"; shift 2 ;;
    --out-dir)     OUT_DIR="${2:?}"; shift 2 ;;
    -h|--help)     sed -n '2,58p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
done

[[ -n "${MODE}" ]] || die "pick one of --generate | --smoke-test | --bench | --calibrate | --patch"
case "${BENCH_CASE}" in
  all|base|tofp|opt) ;;
  *) die "--case must be one of: all base tofp opt (got '${BENCH_CASE}')" ;;
esac

# ============================================================ patch
# Apply the compiler-side optimization patch to an upstream triton checkout.
# No GPU / triton import needed, so handle it before the python/preflight logic.
# Default target is REPO_ROOT — tma-update/'s parent — which is the upstream
# repo root when this folder is copied there.
if [[ "${MODE}" == "patch" ]]; then
  APPLY="${SCRIPT_DIR}/apply_pass_patch.sh"
  [[ -x "${APPLY}" ]] || die "helper not found/executable: ${APPLY}"
  ROOT="${TRITON_ROOT:-${REPO_ROOT}}"
  echo "[run_benchmark] patch: target triton root = ${ROOT}" >&2
  exec "${APPLY}" --triton-root "${ROOT}" "${PATCH_ARGS[@]}"
fi

# Assemble the workload-size args passed through to paged_kv_load.py.
SIZE_ARGS=()
[[ -n "${NUM_KV_BLOCKS}" ]] && SIZE_ARGS+=(--num-kv-blocks "${NUM_KV_BLOCKS}")
[[ -n "${NUM_SEQS}"      ]] && SIZE_ARGS+=(--num-seqs "${NUM_SEQS}")

# --- locate a python with triton importable ---
if [[ -z "${PYTHON_BIN}" ]]; then
  for cand in "${CONDA_PREFIX:-}/bin/python" python3 python; do
    [[ -z "${cand}" ]] && continue
    if "${cand}" -c "import triton" >/dev/null 2>&1; then PYTHON_BIN="${cand}"; break; fi
  done
fi
[[ -n "${PYTHON_BIN}" ]] || die "no python with 'triton' importable; pass --python or set TMA_UPDATE_PYTHON.
  See 'Prerequisites' in tma-update/ptx-opt-test/README.md — the usual cause is
  that torch's bundled triton wheel is being imported instead of your build."
echo "[run_benchmark] python=${PYTHON_BIN} mode=${MODE} arch=${ARCH}" >&2

# --- preflight: catch the env problems we actually hit, with fix hints -------
# 'generate' is offline (no torch / GPU needed); smoke+bench need the full env.
if [[ "${MODE}" != "generate" ]]; then
  "${PYTHON_BIN}" - <<'PY' || die "preflight failed — see message above (prereqs: tma-update/ptx-opt-test/README.md)"
import sys

def fail(msg):
    print(f"[preflight] FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

import os
try:
    import triton
except Exception as e:
    fail(f"cannot import triton ({e})")

# The 3.2.0-wheel-shadowing trap: no set_allocator => too old / wrong triton.
if not hasattr(triton, "set_allocator"):
    fail(f"triton {getattr(triton,'__version__','?')} at {triton.__file__} lacks "
         "set_allocator — this is torch's bundled 3.2.0 wheel, not a TMA-capable "
         "build. Fix: `pip install -e .` from your triton checkout (torch FIRST, "
         "editable triton LAST).")

try:
    import torch
except Exception as e:
    fail(f"cannot import torch ({e}). Install it FIRST, then re-run "
         "`pip install -e .` in your triton checkout. See README Prerequisites.")

if not torch.cuda.is_available():
    fail("torch.cuda.is_available() is False — no NVIDIA GPU visible. smoke/bench "
         "need a Hopper (sm_90) or Blackwell (sm_100) GPU.")

cc = torch.cuda.get_device_capability()
print(f"[preflight] OK: triton {triton.__version__} @ "
      f"{os.path.realpath(triton.__file__)}", file=sys.stderr)
print(f"[preflight] OK: torch {torch.__version__}, GPU sm_{cc[0]}{cc[1]} "
      f"({torch.cuda.get_device_name(0)})", file=sys.stderr)
if cc[0] < 9:
    print(f"[preflight] WARN: sm_{cc[0]}{cc[1]} has no TMA — 'base'/'opt' cases will "
          "not exercise real TMA (opt's sm_100 PTX will be rejected by ptxas).",
          file=sys.stderr)
PY
fi

cd "${REPO_ROOT}"

# ============================================================ generate
if [[ "${MODE}" == "generate" ]]; then
  mkdir -p "${OUT_DIR}"
  GEN_ARGS=()
  [[ -n "${NUM_KV_BLOCKS}" ]] && GEN_ARGS+=(--num-kv-blocks "${NUM_KV_BLOCKS}")
  echo "[run_benchmark] compiling ${KERNEL_NAME} -> PTX (${ARCH}) into ${OUT_DIR}" >&2
  "${PYTHON_BIN}" "${PY_SCRIPT}" compile --arch "${ARCH}" --stop-after ptx \
      --out-dir "${OUT_DIR}" "${GEN_ARGS[@]}"
  echo "[run_benchmark] baseline PTX: ${OUT_DIR}/${KERNEL_NAME}.ptx" >&2
  exit 0
fi

# ============================================================ smoke
if [[ "${MODE}" == "smoke" ]]; then
  echo "[run_benchmark] smoke test (launch + verify vs torch reference)" >&2
  "${PYTHON_BIN}" "${PY_SCRIPT}" smoke
  exit $?
fi

# ============================================================ calibrate
if [[ "${MODE}" == "calibrate" ]]; then
  # Which case to calibrate on: 'all' has no single meaning here -> use base.
  CAL_CASE="${BENCH_CASE}"
  [[ "${CAL_CASE}" == "all" ]] && CAL_CASE="base"
  # map wrapper name -> python case name + per-case env
  PYCASE="tma"
  EXTRA_ENV=()
  case "${CAL_CASE}" in
    base) PYCASE="tma"; EXTRA_ENV+=("TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1") ;;
    tofp) PYCASE="pointer" ;;
    opt)  PYCASE="tma" ;;   # default path: gate pass runs
  esac
  echo "[run_benchmark] calibrating variance on case '${CAL_CASE}' (burst=${BURST})" >&2
  env "${EXTRA_ENV[@]}" TRITON_ALWAYS_COMPILE=1 \
    "${PYTHON_BIN}" "${PY_SCRIPT}" calibrate --case "${PYCASE}" --burst "${BURST}" "${SIZE_ARGS[@]}"
  exit $?
fi

# ============================================================ bench
# Three cases, all default JIT compilation (no PTX override — the optimization
# is now the `triton-rewrite-loop-recreated-descriptors` compiler pass):
#   base  #1  TMA, optimization OFF  -> raw tensormap_create per iteration
#             (TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1 suppresses the pass)
#   tofp  #2  manual tensor-of-pointers kernel (no descriptor)
#   opt   #3  TMA, optimization ON  -> the gate pass demotes the loop-recreated
#             descriptor to pointers (default Hopper+ behavior)
# --case all (default) runs all three + a comparison table; a single name runs
# just that one.
if [[ "${MODE}" == "bench" ]]; then
  RESULTS="$(mktemp)"           # "label<TAB>median_ms"
  grab_median() { grep -oE 'median [0-9.]+' "$1" | head -1 | awk '{print $2}'; }
  want() { [[ "${BENCH_CASE}" == "all" || "${BENCH_CASE}" == "$1" ]]; }

  run_case() {  # $1=label  $2=outfile  (remaining = env + py args)
    local label="$1" out="$2"; shift 2
    echo "" >&2
    echo "=========== ${label} ===========" >&2
    "$@" | tee "${out}"
    local med; med="$(grab_median "${out}")"
    printf '%s\t%s\n' "${label}" "${med:-NA}" >> "${RESULTS}"
  }

  # --- #1 base: TMA with the optimization DISABLED (raw create/iter) --------
  # The disable knob is set in this subprocess so the NVIDIA make_ttir gate
  # skips the loop-recreated-descriptor rewrite.
  if want base; then
    run_case "1. baseline TMA (opt OFF)" /tmp/tma_c1.out \
      env TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE=1 TRITON_ALWAYS_COMPILE=1 \
          "${PYTHON_BIN}" "${PY_SCRIPT}" bench \
          --case tma --warmup "${WARMUP}" --trials "${TRIALS}" "${SIZE_ARGS[@]}"
  fi

  # --- #2 tofp: explicit tl.load, no descriptor -----------------------------
  if want tofp; then
    run_case "2. manual pointer" /tmp/tma_c2.out \
      env TRITON_ALWAYS_COMPILE=1 "${PYTHON_BIN}" "${PY_SCRIPT}" bench \
          --case pointer --warmup "${WARMUP}" --trials "${TRIALS}" "${SIZE_ARGS[@]}"
  fi

  # --- #3 opt: TMA with the optimization ENABLED (default Hopper+ path) ------
  # Same kernel as #1; the gate pass runs by default and demotes the
  # loop-recreated descriptor to pointers.
  if want opt; then
    run_case "3. TMA (opt ON)" /tmp/tma_c3.out \
      env TRITON_ALWAYS_COMPILE=1 "${PYTHON_BIN}" "${PY_SCRIPT}" bench \
          --case tma --warmup "${WARMUP}" --trials "${TRIALS}" "${SIZE_ARGS[@]}"
  fi

  # --- summary table (only meaningful when >1 case ran) ---------------------
  if [[ "${BENCH_CASE}" == "all" ]]; then
    echo "" >&2
    echo "================= 3-CASE SUMMARY (median ms) =================" >&2
    "${PYTHON_BIN}" - "${RESULTS}" <<'PY'
import sys
rows = [ln.rstrip("\n").split("\t") for ln in open(sys.argv[1]) if ln.strip()]
vals = {}
for label, med in rows:
    print(f"  {label:<28} {med:>10} ms")
    try: vals[label] = float(med)
    except ValueError: pass
print()
def cmp(a, b):
    if a in vals and b in vals and vals[b] > 0:
        r = vals[a] / vals[b]
        print(f"  {a.split('.')[0]} vs {b.split('.')[0]}: "
              f"{r:.3f}x  ({(vals[a]-vals[b])/vals[a]*100:+.1f}%)")
keys = list(vals)
def find(n): return next((k for k in keys if k.startswith(n)), None)
c1,c2,c3 = (find("1"),find("2"),find("3"))
if c1 and c3: cmp(c1,c3)   # does the optimization help the TMA path?
if c1 and c2: cmp(c1,c2)   # baseline TMA vs manual pointers
if c3 and c2: cmp(c3,c2)   # optimized TMA vs manual pointers (should be ~parity)
PY
  else
    echo "" >&2
    echo "[run_benchmark] case '${BENCH_CASE}': $(grab_median "$(ls -t /tmp/tma_c*.out | head -1)") ms" >&2
  fi
  exit 0
fi
