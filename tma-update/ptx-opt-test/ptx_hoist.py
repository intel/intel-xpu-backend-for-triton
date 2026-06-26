#!/usr/bin/env python3
"""
ptx_hoist.py — hoist the loop-invariant TMA descriptor build out of a Triton
`tensormap_create`-per-iteration loop, leaving only the `global_address` replace
(+ cp_fenceproxy + acquire) in the loop body.

This scripts the hand edit documented in NOTES.md / ANNOTATED.md so the A/B
optimized PTX is reproducible and reviewable instead of hand-maintained.

WHAT IT DOES (single-buffer pattern, e.g. paged_kv_load_kernel):
  1. Split the PTX into prologue / loop-body / epilogue around the loop label
     and its back-edge.
  2. In the loop body, find the descriptor build:
       - the zero-fill (st.shared + bracketing bar.sync / bar.warp.sync),
       - the `cvt.u64.u32 %rdDESC, %rSMEM` that names the descriptor SMEM addr,
       - every `tensormap.replace.tile.<field>` inline-asm block.
  3. Classify each replace: `global_address` stays in the loop (it varies);
     all others are invariant -> hoist.
  4. Relocate the descriptor SMEM staging to a non-overlapping, PERSISTENT
     offset (default +4224) so it survives across iterations, and grow the
     b32 register file by one for the relocated-base register.
  5. Emit the invariant build ONCE before the loop label; keep only the
     global_address replace + cp_fenceproxy + acquire in the loop.

It is deliberately CONSERVATIVE: every structural assumption is asserted. If the
input PTX does not match the expected shape it raises with a clear message
rather than emitting wrong PTX. It is scoped to the single-buffer pattern; the
pipelined (ring-slot) case is out of scope (see bench-plan.md phase 2).

Usage:
  ptx_hoist.py IN.ptx OUT.ptx [--reloc-offset N] [--check-against FILE]
"""

import argparse
import re
import sys

# ----------------------------------------------------------------------------
# Chunking: an inline-asm block (begin..end) is one chunk; any other line is its
# own chunk. This lets us move whole `tensormap.replace.tile.*` blocks atomically.
# ----------------------------------------------------------------------------


class Chunk:
    __slots__ = ("lines", "is_asm")

    def __init__(self, lines, is_asm):
        self.lines = lines  # list[str] (no trailing newline)
        self.is_asm = is_asm

    def text(self):
        return "\n".join(self.lines)


def chunk_lines(lines):
    """Group lines into chunks, coalescing `// begin inline asm`..`// end inline asm`."""
    chunks = []
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].strip() == "// begin inline asm":
            j = i
            while j < n and lines[j].strip() != "// end inline asm":
                j += 1
            assert j < n, "unterminated inline asm block"
            chunks.append(Chunk(lines[i:j + 1], is_asm=True))
            i = j + 1
        else:
            chunks.append(Chunk([lines[i]], is_asm=False))
            i += 1
    return chunks


def chunk_has(chunk, needle):
    return any(needle in ln for ln in chunk.lines)


# ----------------------------------------------------------------------------
# Transform
# ----------------------------------------------------------------------------

REPLACE_RE = re.compile(r"tensormap\.replace\.tile\.([a-z_]+)\.")


def transform(text, reloc_offset):
    lines = text.split("\n")

    # --- 1. locate the loop label and its back-edge ---------------------------
    label_idx = next((k for k, ln in enumerate(lines) if re.match(r"^\$L__BB\d+_\d+:", ln.strip())), None)
    assert label_idx is not None, "could not find a loop label ($L__BB*_*:)"
    label = lines[label_idx].strip().split(":")[0]

    backedge_idx = next(
        (k for k in range(label_idx + 1, len(lines)) if f"bra \t{label};" in lines[k] or f"bra {label};" in lines[k]),
        None)
    assert backedge_idx is not None, f"could not find back-edge 'bra {label};'"

    prologue = lines[:label_idx]
    body = lines[label_idx + 1:backedge_idx + 1]  # includes the bra line
    epilogue = lines[backedge_idx + 1:]

    # --- 2. find the descriptor SMEM base register ---------------------------
    # The loop contains `cvt.u64.u32 %rdDESC, %rSMEM` feeding the tensormap ops.
    cvt_re = re.compile(r"cvt\.u64\.u32\s+(%rd\d+),\s+(%r\d+);")
    desc_addr_reg = None  # %rdDESC
    smem_base_reg = None  # %rSMEM
    for ln in body:
        m = cvt_re.search(ln)
        if m and any("tensormap.replace" in b for b in body):
            desc_addr_reg, smem_base_reg = m.group(1), m.group(2)
            break
    assert desc_addr_reg and smem_base_reg, \
        "could not find `cvt.u64.u32 %rdDESC, %rSMEM` (descriptor addr) in loop"

    # --- 3. find the zero-fill dest register ---------------------------------
    # `@%pX st.shared::cta.b32 [ %rZF + 0 ], %rVAL;`
    zf_re = re.compile(r"st\.shared::cta\.b32\s+\[\s+(%r\d+)\s+\+\s+0\s+\],\s+(%r\d+);")
    zf_dest_reg = None
    for ln in body:
        m = zf_re.search(ln)
        if m:
            zf_dest_reg = m.group(1)
            break
    assert zf_dest_reg, "could not find zero-fill `st.shared::cta.b32 [ %rZF + 0 ], ...`"

    # The prologue defines zf_dest_reg as `add.s32 %rZF, %rSMEM, %rOFF;`.
    zf_def_re = re.compile(rf"^\s*add\.s32\s+{re.escape(zf_dest_reg)},\s+{re.escape(smem_base_reg)},\s+(%r\d+);")
    zf_def_idx = next((k for k, ln in enumerate(prologue) if zf_def_re.match(ln)), None)
    assert zf_def_idx is not None, \
        f"could not find prologue def `add.s32 {zf_dest_reg}, {smem_base_reg}, %rOFF;`"
    zf_off_reg = zf_def_re.match(prologue[zf_def_idx]).group(1)

    # --- 4. allocate a fresh b32 register for the relocated base -------------
    reg_decl_re = re.compile(r"^(\s*\.reg\s+\.b32\s+%r)<(\d+)>;")
    reg_decl_idx = next((k for k, ln in enumerate(prologue) if reg_decl_re.match(ln)), None)
    assert reg_decl_idx is not None, "could not find `.reg .b32 %r<N>;` decl"
    m = reg_decl_re.match(prologue[reg_decl_idx])
    reg_count = int(m.group(2))
    reloc_base_reg = f"%r{reg_count}"  # first free index
    prologue[reg_decl_idx] = f"{m.group(1)}<{reg_count + 1}>;"

    # --- 5. rewrite prologue: relocate base, repoint zero-fill dest -----------
    # Insert `add.s32 %rRELOC, %rSMEM, OFFSET;` right after the smem-base mov.
    smem_mov_re = re.compile(rf"^\s*mov\.b32\s+{re.escape(smem_base_reg)},\s+global_smem;")
    smem_mov_idx = next((k for k, ln in enumerate(prologue) if smem_mov_re.match(ln)), None)
    assert smem_mov_idx is not None, \
        f"could not find `mov.b32 {smem_base_reg}, global_smem;`"
    indent = "\t"
    prologue.insert(
        smem_mov_idx + 1, f"{indent}add.s32 \t{reloc_base_reg}, {smem_base_reg}, {reloc_offset};"
        f"   // HOIST: relocate descriptor staging (persistent, non-overlapping)")
    # zf_def_idx may have shifted by the insert above:
    if zf_def_idx > smem_mov_idx:
        zf_def_idx += 1
    # repoint zero-fill dest at the relocated base
    prologue[zf_def_idx] = (f"{indent}add.s32 \t{zf_dest_reg}, {reloc_base_reg}, {zf_off_reg};"
                            f"   // HOIST: zero-fill into relocated staging")

    # --- 6. walk the loop body in chunks, classify ----------------------------
    chunks = chunk_lines(body)

    hoist = []  # invariant build, emitted once before the loop
    keep = []  # remains in the loop
    invariant_replaces = 0
    kept_global_addr = 0
    saw_zero_fill = saw_cvt = False

    for ch in chunks:
        # the descriptor-addr cvt: relocate + hoist
        if (not ch.is_asm) and cvt_re.search(ch.lines[0]) \
                and desc_addr_reg in ch.lines[0]:
            hoist.append(
                Chunk([
                    re.sub(re.escape(smem_base_reg) + r"\b", reloc_base_reg, ch.lines[0]) +
                    "   // HOIST: descriptor addr = relocated base"
                ], False))
            saw_cvt = True
            continue

        # zero-fill inline-asm block + its bracketing barriers/mov: hoist
        if ch.is_asm and chunk_has(ch, "st.shared::cta.b32"):
            hoist.append(ch)
            saw_zero_fill = True
            continue

        # tensormap.replace blocks: global_address stays, others hoist
        if ch.is_asm and chunk_has(ch, "tensormap.replace.tile"):
            field = None
            for ln in ch.lines:
                m = REPLACE_RE.search(ln)
                if m:
                    field = m.group(1)
                    break
            if field == "global_address":
                keep.append(ch)
                kept_global_addr += 1
            else:
                hoist.append(ch)
                invariant_replaces += 1
            continue

        # bracketing barriers / scalar movs that feed the build: hoist a copy.
        # `bar.sync 0`, `bar.warp.sync -1`, `mov.b32 %rX, IMM;` immediately
        # surrounding the zero-fill are part of the one-time build.
        s = ch.lines[0].strip() if not ch.is_asm else ""
        if s in ("bar.sync \t0;", "bar.warp.sync \t-1;") and not saw_cvt:
            # these precede the cvt/zerofill -> part of the build prologue
            hoist.append(ch)
            continue
        if not ch.is_asm and re.match(r"^\s*mov\.b32\s+%r\d+,\s+0;", ch.lines[0]) \
                and not saw_cvt:
            hoist.append(ch)  # zero-fill value (%rVAL = 0); safe (constant)
            keep.append(ch)  # also keep: reused later as a coord/parity reg
            continue
        if not ch.is_asm and re.match(r"^\s*mov\.b32\s+%r\d+,\s+\d+;", ch.lines[0]) \
                and not saw_cvt is False:
            # constant movs feeding invariant replaces (e.g. box_dim=64): hoist
            # (they sit between the global_address keep and the invariant ones)
            hoist.append(ch)
            continue

        keep.append(ch)

    assert saw_zero_fill, "did not find the zero-fill block in the loop"
    assert saw_cvt, "did not find the descriptor-addr cvt in the loop"
    assert kept_global_addr == 1, \
        f"expected exactly 1 global_address replace in loop, found {kept_global_addr}"
    assert invariant_replaces >= 1, "found no invariant replaces to hoist"

    # --- 7. reassemble --------------------------------------------------------
    out = []
    out.extend(prologue)
    out.append("")
    out.append(f"{indent}// ===== HOISTED loop-invariant TMA descriptor build (runs ONCE) =====")
    for ch in hoist:
        out.extend(ch.lines)
    out.append(f"{indent}// ===== end hoisted build =====")
    out.append("")
    out.append(f"{label}:                              // =>This Inner Loop Header: Depth=1")
    for ch in keep:
        out.extend(ch.lines)
    out.extend(epilogue)

    summary = dict(label=label, desc_addr_reg=desc_addr_reg, smem_base_reg=smem_base_reg, reloc_base_reg=reloc_base_reg,
                   reloc_offset=reloc_offset, invariant_replaces=invariant_replaces, zf_dest_reg=zf_dest_reg)
    return "\n".join(out), summary


def count_loop_replaces(text):
    """Count tensormap.replace.tile.* inside the loop body (for reporting)."""
    lines = text.split("\n")
    li = next((k for k, ln in enumerate(lines) if re.match(r"^\$L__BB\d+_\d+:", ln.strip())), None)
    if li is None:
        return None
    label = lines[li].strip().split(":")[0]
    be = next((k for k in range(li + 1, len(lines)) if f"bra \t{label};" in lines[k] or f"bra {label};" in lines[k]),
              len(lines))
    return sum(1 for ln in lines[li:be] if "tensormap.replace.tile" in ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="baseline PTX")
    ap.add_argument("output", help="optimized PTX to write")
    ap.add_argument(
        "--reloc-offset", type=int, default=4224, help="SMEM byte offset for the persistent descriptor staging "
        "(default 4224 = past the 4096B data tile + mbarrier). "
        "MUST NOT overlap any other SMEM use — verify for your kernel.")
    ap.add_argument(
        "--check-against", default=None, help="compare loop-body replace count against this reference "
        "optimized PTX (e.g. the hand edit) and warn on mismatch")
    args = ap.parse_args()

    with open(args.input) as f:
        src = f.read()

    before = count_loop_replaces(src)
    out, summary = transform(src, args.reloc_offset)
    after = count_loop_replaces(out)

    with open(args.output, "w") as f:
        f.write(out)

    print(f"[ptx_hoist] loop label        : {summary['label']}", file=sys.stderr)
    print(
        f"[ptx_hoist] descriptor addr   : {summary['desc_addr_reg']} "
        f"(was {summary['smem_base_reg']}, relocated to {summary['reloc_base_reg']} "
        f"= global_smem+{summary['reloc_offset']})", file=sys.stderr)
    print(f"[ptx_hoist] invariant replaces hoisted : {summary['invariant_replaces']}", file=sys.stderr)
    print(f"[ptx_hoist] loop-body replaces : {before} -> {after}", file=sys.stderr)
    print(f"[ptx_hoist] wrote {args.output}", file=sys.stderr)

    if after != 1:
        print(f"[ptx_hoist] WARNING: expected 1 replace left in loop, got {after}", file=sys.stderr)
        return 1

    if args.check_against:
        ref_after = count_loop_replaces(open(args.check_against).read())
        if ref_after != after:
            print(
                f"[ptx_hoist] WARNING: loop replace count {after} != reference "
                f"{ref_after} ({args.check_against})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
