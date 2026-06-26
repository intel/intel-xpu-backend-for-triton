#!/usr/bin/env python3
"""
gen_knob_patch.py — (re)generate the TRITON_INTEL_NVIDIA_FORCE_TD_TEST knob patch
against a target Triton repo.

The #3 "TD->pointer fallback" benchmark case needs a debug knob that widens the
NVIDIA `make_ttir` gate so the `make_tensor_descriptor` -> pointer rewrite runs
on TMA-capable targets. That touches two files OUTSIDE tma-update/:

  * python/triton/knobs.py                     (declare the knob)
  * third_party/nvidia/backend/compiler.py     (widen the gate)

A committed static diff would fail to apply if upstream changed those files.
Instead this script reads the TARGET repo's CURRENT files, applies the two edits
via SEMANTIC anchors (class name; the rewrite-pass call), and emits a unified
diff generated FROM the target — so it always applies cleanly there.

Idempotent: if the knob is already present it contributes nothing.

Usage:
  gen_knob_patch.py --triton-root /path/to/triton         # emit diff to stdout
  gen_knob_patch.py --triton-root /path/to/triton --out knob.patch
  gen_knob_patch.py --triton-root /path/to/triton --apply  # edit files in place
  gen_knob_patch.py                                        # auto-detect installed triton

Apply the emitted patch later with:
  git apply -p1 knob.patch      (run from the triton repo root)
"""

import argparse
import difflib
import os
import sys

ENV_NAME = "TRITON_INTEL_NVIDIA_FORCE_TD_TEST"
FIELD_NAME = "nvidia_force_td_test"

KNOB_BLOCK = [
    "    # Debug/benchmark only: on TMA-capable NVIDIA targets (sm>=9.0), force the",
    "    # `make_tensor_descriptor` -> pointer rewrite that is normally reserved for",
    "    # pre-Hopper. Lets us measure the non-TMA fallback path on Hopper/Blackwell.",
    "    # NVIDIA-only — other backends select descriptor lowering differently.",
    f'    {FIELD_NAME}: env_bool = env_bool("{ENV_NAME}")',
]

GATE_ADDITION = f" or knobs.compilation.{FIELD_NAME}"

# ----------------------------------------------------------------------------
# file location
# ----------------------------------------------------------------------------


def _candidates(root, rels):
    return [os.path.join(root, *rel.split("/")) for rel in rels]


def locate_files(triton_root):
    """
    Return (knobs_path, compiler_path, repo_root_for_diff_paths).

    Prefers a source checkout layout under triton_root; falls back to the
    installed `triton` package.
    """
    if triton_root:
        knobs_cands = _candidates(triton_root, [
            "python/triton/knobs.py",
            "triton/knobs.py",
            "knobs.py",
        ])
        comp_cands = _candidates(triton_root, [
            "third_party/nvidia/backend/compiler.py",
            "python/triton/backends/nvidia/compiler.py",
            "triton/backends/nvidia/compiler.py",
        ])
        knobs_path = next((c for c in knobs_cands if os.path.isfile(c)), None)
        comp_path = next((c for c in comp_cands if os.path.isfile(c)), None)
        if not knobs_path or not comp_path:
            missing = "knobs.py" if not knobs_path else "nvidia/backend/compiler.py"
            raise SystemExit(f"[gen_knob_patch] could not find {missing} under {triton_root}\n"
                             f"  searched: {knobs_cands if not knobs_path else comp_cands}")
        return knobs_path, comp_path, triton_root

    # auto-detect installed triton
    try:
        import triton
    except ImportError:
        raise SystemExit("[gen_knob_patch] triton not importable and no --triton-root given")
    pkg = os.path.dirname(triton.__file__)
    knobs_path = os.path.join(pkg, "knobs.py")
    comp_path = os.path.join(pkg, "backends", "nvidia", "compiler.py")
    for p in (knobs_path, comp_path):
        if not os.path.isfile(p):
            raise SystemExit(f"[gen_knob_patch] expected file not found: {p}")
    # diff paths relative to the package parent so they look sane
    return knobs_path, comp_path, os.path.dirname(pkg)


# ----------------------------------------------------------------------------
# edits (anchored, idempotent)
# ----------------------------------------------------------------------------


def edit_knobs(text):
    """Insert the knob field as the first member of class compilation_knobs."""
    if FIELD_NAME in text or ENV_NAME in text:
        return text, False  # already applied
    lines = text.split("\n")
    idx = next((k for k, ln in enumerate(lines) if ln.strip() == "class compilation_knobs(base_knobs):"), None)
    if idx is None:
        raise SystemExit("[gen_knob_patch] anchor not found in knobs.py: "
                         "'class compilation_knobs(base_knobs):'")
    # insert right after the class declaration line
    new = lines[:idx + 1] + KNOB_BLOCK + lines[idx + 1:]
    return "\n".join(new), True


def edit_compiler(text):
    """
    Widen the pre-Hopper gate that guards add_rewrite_tensor_descriptor_to_pointer.

    Anchors on the pass call, then walks up to the nearest enclosing
    `if ... capability ... < 9 ...:` and appends the knob to its condition.
    """
    if FIELD_NAME in text:
        return text, False  # already applied
    lines = text.split("\n")
    call_idx = next((k for k, ln in enumerate(lines) if "add_rewrite_tensor_descriptor_to_pointer" in ln), None)
    if call_idx is None:
        raise SystemExit("[gen_knob_patch] anchor not found in compiler.py: "
                         "'add_rewrite_tensor_descriptor_to_pointer'")
    gate_idx = next(
        (k for k in range(call_idx - 1, -1, -1)
         if "if " in lines[k] and "capability" in lines[k] and "< 9" in lines[k] and lines[k].rstrip().endswith(":")),
        None)
    if gate_idx is None:
        raise SystemExit("[gen_knob_patch] could not find the enclosing "
                         "'if capability // 10 < 9:' gate above the rewrite call")
    ln = lines[gate_idx]
    # append the knob before the trailing ':'  (preserve indentation + condition)
    stripped = ln.rstrip()
    assert stripped.endswith(":")
    lines[gate_idx] = stripped[:-1] + GATE_ADDITION + ":"

    # sanity: knobs must be importable in this module
    if "import knobs" not in text and "from triton import knobs" not in text:
        print(
            "[gen_knob_patch] WARNING: 'knobs' does not appear to be imported in "
            "compiler.py — the gate edit will NameError until you add "
            "'from triton import knobs'.", file=sys.stderr)
    return "\n".join(lines), True


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------


def unified(path, before, after, root):
    rel = os.path.relpath(path, root)
    return difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile=f"a/{rel}",
                                tofile=f"b/{rel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triton-root", default=None,
                    help="root of the target Triton repo (default: auto-detect installed triton)")
    ap.add_argument("--out", default=None, help="write the combined patch here (default: stdout)")
    ap.add_argument("--apply", action="store_true", help="edit the target files in place")
    args = ap.parse_args()

    knobs_path, comp_path, root = locate_files(args.triton_root)

    knobs_before = open(knobs_path).read()
    comp_before = open(comp_path).read()
    knobs_after, k_changed = edit_knobs(knobs_before)
    comp_after, c_changed = edit_compiler(comp_before)

    print(f"[gen_knob_patch] knobs.py    : {knobs_path} "
          f"({'edit' if k_changed else 'already applied'})", file=sys.stderr)
    print(f"[gen_knob_patch] compiler.py : {comp_path} "
          f"({'edit' if c_changed else 'already applied'})", file=sys.stderr)

    if not (k_changed or c_changed):
        print("[gen_knob_patch] nothing to do — knob already present.", file=sys.stderr)
        return 0

    if args.apply:
        if k_changed:
            open(knobs_path, "w").write(knobs_after)
        if c_changed:
            open(comp_path, "w").write(comp_after)
        print("[gen_knob_patch] applied edits in place.", file=sys.stderr)
        return 0

    diff = list(unified(knobs_path, knobs_before, knobs_after, root)) \
        + list(unified(comp_path, comp_before, comp_after, root))
    patch_text = "".join(diff)
    if args.out:
        open(args.out, "w").write(patch_text)
        print(f"[gen_knob_patch] wrote patch to {args.out}  "
              f"(apply with: git apply -p1 {args.out})", file=sys.stderr)
    else:
        sys.stdout.write(patch_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
