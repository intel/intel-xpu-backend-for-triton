# Performance Notes — Why Migrate, and How to Stay on the Fast Path

All claims here are verified against the Intel XPU backend source. Read this to
write the per-descriptor performance note and to honor the prime directive
(produce an *efficiently-lowerable* descriptor).

## How device-side descriptors actually compile on XPU

`tl.make_block_ptr` is deprecated, but block pointers and tensor descriptors do
**not** share a single internal pass that makes them identical. The verified
pipeline:

- **`RewriteTensorDescriptorToPointer`** (`compiler.py`, TTIR stage) does **not**
  blindly erase all descriptors. It builds a `candidateMakeTensorDescOps` set and
  marks those **legal = preserved**. A descriptor feeding a
  `DescriptorLoadOp`/`StoreOp` that **traces cleanly to an in-kernel
  `MakeTensorDescOp`** is kept; it survives to TTGIR and is lowered to **fast
  hardware 2D block I/O** by `LowerTo2DBlockLoad` (`compiler.py`, later TTGIR
  stage). A descriptor that cannot be traced (function argument, `tt.call`
  result, some conditionals) or feeds a gather/scatter is **rewritten to
  pointer+mask** (the slower indexed path).
- **Block pointers** are handled by `MaterializeBlockPointer`, which sets
  `ttig.block_io` and reaches the same 2D-block-I/O hardware path.

**Consequence:** a well-formed, locally-created, 2D, dot-feeding descriptor
reaches the **same** 2D-block-I/O path the block pointer used. Migration is
primarily **API modernization / removing deprecation** — at worst neutral, and
it keeps the fast path. A *naive* translation, however, can trip a gate and fall
to the slower pointer path. That is the regression the skill must prevent.

> Do **not** promise ">2x". The README's ">2x" figure is vs the
> **tensor-of-pointers** approach (manual `tl.arange` + pointer arithmetic), a
> slower starting point than block pointers.

## The gates for fast 2D block I/O (verified, current C++)

A descriptor load reaches 2D block I/O only when **all** hold:

1. **Device supports it** — module attribute `support_2d_block_io` (PVC, BMG;
   **not** DG2).
2. **Rank ≥ 2** — `MaterializeBlockPointer` returns early at rank 1, so a **1D
   descriptor does NOT get block I/O today** (the 1D fast path is pending).
3. **Traceable to a local `MakeTensorDescOp`** — not passed as a function arg or
   across `tt.call`; consistent padding across descriptor candidates.
4. **Feeds a `tl.dot`** — the result carries DPAS / DotOperand(DPAS) encoding,
   unless `TRITON_INTEL_ENABLE_BLOCK_IO_ALL_LAYOUTS` is set. A standalone load
   not feeding a dot will not get block I/O by default.
5. **Valid tile** from the LinearLayout (`validate2DBlockLoadTile`).
6. **Rank-3+ batch dim folded out** — row/col tile dims must be the inner two
   dims (fold the batch into `base`, Rule 8/11).
7. **OWord pitch** — leading stride (pitch) must be ≥ 64 bytes and divisible by
   `ceil(128 / elem_bits)` (stricter than the block-ptr path's 16-byte rule).
8. **Transpose only for 32/64-bit** element types.

## Translation decisions that keep the fast path

The skill's rules are written so the *default* output satisfies the gates:

- **Last stride 1** (Rule 6): express transposes with `.T` on the loaded block,
  never a descriptor with last stride ≠ 1 (would fail tile/pitch gates).
- **2D** (Rule 11): fold rank>2 into `base`; never emit a 3D descriptor for a
  batched access.
- **Local** (Rule 12): build the descriptor in the function that loads from it,
  never thread it through `tt.call`.
- **PAD_ZERO** (Rule 3): keep the default; only carry `"nan"` when the source
  demanded it.
- **Dot-feeding** (gate 4): for a load that does not feed a `tl.dot`, say so.

## Regression risks to flag (when an efficient form isn't possible)

Translate anyway (the user wants the modern API), but flag the reason:

- **Untraceable descriptor** — built in a helper and passed across `tt.call`
  (Rule 12 fixes this; if the caller can't be changed, flag it).
- **Non-OWord pitch** — a leading stride that was fine for block-ptr block I/O
  (16-byte) but isn't 128-bit aligned drops a descriptor to the pointer path.
- **Not feeding a dot** — no DPAS encoding ⇒ pointer path unless the env override
  is set.
- **rank>2 not foldable** — an outer index that is a vector/range (not a
  per-program scalar) cannot fold into `base`.
- **rank-1** — no block I/O today.

## No host-side allocator on XPU (verified)

Device-side descriptors need **no** `triton.set_allocator` and **no** host-launch
change on XPU:

- The 2D-block-I/O path (`LowerTo2DBlockLoad`) issues hardware block-load
  messages that read global memory directly — no global scratch.
- The pointer-rewrite path is ordinary pointer+mask loads — no scratch.
- `allocate_global_scratch_memory` finds nothing descriptor-related to size.
- The repo's own descriptor benchmarks (`gemm_benchmark.py`,
  `flash_attention_benchmark.py`, `gemm_streamk_benchmark.py`) never call
  `set_allocator`.
- `triton.set_allocator` is **NVIDIA-TMA-specific** (TMA hardware descriptors
  need a workspace).

So the skill never emits an allocator or any host change. The migration is a
pure kernel-body rewrite (except Rule 12, which changes a `@triton.jit` helper
signature and its call sites — still no host/launch change).

## How to confirm the fast path for a given kernel

Dump IR and look for a 2D block load (see the `ir-debugging` skill):
```
MLIR_ENABLE_DUMP=1 TRITON_ALWAYS_COMPILE=1 python your_kernel.py 2>&1 | grep -iE "2dblock|block_io|Subgroup2DBlock"
```
If you see a 2D block load for the descriptor's load, it's on the fast path. If
you see scalarized pointer loads / a gather instead, a gate was tripped — revisit
the decisions above.
