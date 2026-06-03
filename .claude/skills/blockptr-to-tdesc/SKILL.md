---
name: blockptr-to-tdesc
description: >-
  Translate a Triton kernel from the deprecated block-pointer API
  (tl.make_block_ptr / tl.advance / tl.load(boundary_check=...)) into an
  equivalent kernel using the modern device-side tensor-descriptor API
  (tl.make_tensor_descriptor / desc.load / desc.store) for the Intel XPU
  backend. Use this skill whenever the user wants to migrate, convert,
  translate, port, modernize, or "update" a kernel from block pointers to
  tensor descriptors; whenever they mention tl.make_block_ptr or tl.advance and
  ask for a modern/non-deprecated equivalent; whenever they ask how to use
  tensor descriptors in a kernel that currently uses block pointers; or when
  they paste a kernel using block pointers and ask how to speed it up or make it
  use DPAS / 2D block I/O on Intel GPU (PVC/BMG). Produce the descriptor form
  the XPU backend can lower efficiently, not just any form that compiles.
---

# Block Pointer → Tensor Descriptor (Intel XPU)

Translate kernels from the **deprecated** block-pointer API to the **device-side
tensor-descriptor** API. `tl.make_block_ptr` is deprecated; its docstring points
users to `tl.make_tensor_descriptor`. On Intel XPU, a well-formed descriptor
lowers to the same hardware 2D-block-I/O path the block pointer used, so a
careful translation is at worst neutral and removes the deprecation.

## Prime directive: translate toward an *efficiently-lowerable* descriptor

Do **not** emit the first descriptor that compiles. There are usually several
faithful ways to write a descriptor, and only some keep the fast 2D-block-I/O
path on XPU. Always aim for a descriptor that is **2D, last-stride-1,
locally-created, dot-feeding, PAD_ZERO**. When that is achievable (the common
GEMM/attention cases), produce it. When a particular load genuinely cannot be
put on the fast path, still translate it (the user wants the modern API) but
**explicitly flag** the caveat in the "Changes made" section so nobody is misled
into thinking every load got block I/O.

See `references/performance-notes.md` for *why* (the verified backend gates) and
`references/api-reference.md` for exact signatures and constraints. See
`references/examples.md` for verbatim before→after pairs you can pattern-match
against.

## Workflow

1. **Read the kernel** (inline or from the given file path). Identify every
   block-pointer construct: `tl.make_block_ptr`, `tl.advance`, and the
   `tl.load`/`tl.store` calls that consume block pointers. Also note any
   interleaved `tl.load(ptrs, mask=...)` tensor-of-pointer loads (Rule 9) and
   the rank of each block pointer (Rule 11).
2. **Apply the transformation rules below**, honoring the prime directive.
3. **Emit the output** in the format specified at the bottom.

This is XPU-only. **Never** add `triton.set_allocator` or any host-launch change
— device-side descriptors need no global-memory workspace on XPU (that hook is
NVIDIA-TMA-specific). The migration is a pure kernel-body rewrite unless Rule 12
applies.

## Transformation rules

1. **`tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)` →
   `tl.make_tensor_descriptor(base, shape, strides, block_shape)`.** Drop
   `offsets` and `order` from the constructor. The initial `offsets` values
   become the starting load/store offsets (Rules 2 & 7). Hoist the descriptor to
   the same scope as the old `make_block_ptr` — it is loop-invariant, create it
   once before the loop.

2. **`tl.advance(ptr, delta)` → integer offset increment.** Delete the
   `tl.advance`. Introduce a running integer for each advancing dimension,
   initialized to that dim's initial offset (`off_k = 0` typically), and add the
   delta at the loop tail (`off_k += BLOCK_K`). Pass the running offset into
   `desc.load([...])` / `desc.store([...])`.

3. **`tl.load(ptr, boundary_check=...)` → `desc.load([d0, d1])`.** Drop
   `boundary_check` — the descriptor's `shape` already masks out-of-bounds
   elements to the pad value, which is exactly what `boundary_check` did.
   **Padding caveat:** `padding_option` is set at *descriptor-creation* time, not
   at load time. If the original load used `padding_option="nan"`, carry it onto
   the constructor: `tl.make_tensor_descriptor(..., padding_option="nan")`.
   `"zero"`/`""` is the default and is dropped.

4. **`tl.store(ptr, val, boundary_check=...)` → `desc.store([d0, d1], val)`.**
   Drop `boundary_check`; out-of-bounds writes are ignored automatically. Store
   has no `padding_option`.

5. **Block-pointer loads/stores never carry `mask=`/`other=`** (the API forbids
   it). Their only masking is `boundary_check`, handled by Rules 3–4. A `mask=`
   on the *input* only appears on legacy *tensor-of-pointer* loads — see Rule 9.
   (The skill itself only *emits* a `mask=` tensor-of-pointer load in the single
   sanctioned case of a non-unit-stride rank-1 block pointer that has no legal
   descriptor — see Rule 11.)

6. **Transpose (`order=(0, 1)` in the block pointer).** Do NOT emit a descriptor
   whose last stride ≠ 1 — that drops off the fast path. Instead describe the
   tensor in its native row-major layout (so the last stride is 1) and apply
   `.T` to the loaded block. Concretely, if B is stored `(N, K)` row-major but
   the dot wants `(K, N)`:
   `b_desc = tl.make_tensor_descriptor(base=b_ptr, shape=(N, K), strides=(stride_bn, stride_bk), block_shape=(BLOCK_N, BLOCK_K))`
   then `b = b_desc.load([pid_n*BLOCK_N, off_k]).T`.

7. **Static offsets** (a dim that never advances, e.g. `pid_m * BLOCK_M`) → pass
   the expression directly in `.load()` / `.store()`; no running variable needed.

8. **Batched / 3D** → pre-add the batch offset to `base`
   (`base = a_ptr + bid.to(tl.int64) * stride_az`) and keep the descriptor **2D**
   over the M×K (or K×N) slice. This is the rank-3 special case of Rule 11.

9. **Masked tensor-of-pointer loads** (`tl.load(ptrs, mask=..., other=...)`, not
   block pointers — may be interleaved in the same kernel). Classify the mask:
   - **Boundary mask** — a range check vs a dimension limit, e.g.
     `offs_k[None,:] < K - k*BLOCK_K` or `offs_m[:,None] < M`. This *is* a bounds
     check: fold it into the descriptor `shape` (e.g. `shape=(M, K)`), drop the
     mask, and let the descriptor zero-pad.
   - **Data-dependent / non-boundary mask** — causal masks, value-dependent
     predicates. Cannot be a rectangular `shape`. Load the full block with
     `desc.load([...])`, then re-apply in registers: `v = tl.where(mask, v, other)`.
   - **Mixed** — split: boundary part → `shape`, residual predicate → `tl.where`.

10. **Type-annotation hygiene** → if the **last** stride is annotated
    `tl.int64`/`tl.int32`, flag it (best practice: never annotate the last
    stride; prefer `tl.constexpr` or no annotation). Do **not** silently change
    a signature.

11. **Rank normalization → always aim for 2D** (XPU optimizes only 2D; block
    pointers may be any rank):
    - **Rank 2** → map 1:1.
    - **Rank ≥ 3** → fold the leading dims into `base` (one term per outer dim,
      `base + z*stride_z + h*stride_h`) and build a 2D descriptor over the inner
      two dims; `.T`/`reshape` the result if the consumer needs it. Each folded
      outer index must be a per-program scalar (e.g. `program_id`, a scalar
      load). If an outer index is itself a vector/range it cannot be folded —
      flag it and keep the original.
    - **Rank 1, unit stride** (`strides=(1,)`) → translate to a 1D descriptor:
      `tl.make_tensor_descriptor(base, shape=(N,), strides=(1,), block_shape=(BLOCK,))`,
      `desc.load([off])` / `desc.store([off], val)`. Note that XPU does **not**
      give 1D descriptors 2D block I/O today (the 1D fast path is pending), so
      flag that this load lowers via the pointer path — the form is correct and
      forward-compatible.
    - **Rank 1, non-unit stride** (a strided 1D view, e.g. SGLang FLA's
      `p_beta` = `tl.make_block_ptr(beta + bos*H + i_h, (T,), (H,), (i_t*BT,), (BT,), (0,))`
      with stride `H != 1`) → **no legal descriptor exists**: a descriptor's last
      (here only) stride must be 1, so `strides=(H,)` fails to compile (`Tensor
      descriptor last dim must be 1 but got H`). This is the one case where a
      `boundary_check` **cannot** be modernized into a descriptor at all — so do
      **not** emit a descriptor. Reproduce the access as an explicit
      tensor-of-pointer load/store whose `mask` re-expresses the original
      `boundary_check` (it stays on the pointer path — no 2D block I/O — which is
      the only correct option here, unlike Rule 9 which *removes* such masks by
      folding them into a descriptor `shape`):
      ```python
      o = i_t * BT + tl.arange(0, BT)
      b_beta = tl.load(beta + bos*H + i_h + o*H, mask=o < T, other=0.0)
      # strided rank-1 store: tl.store(g + bos*H + i_h + o*H, b_val, mask=o < T)
      ```
      If the offset advances in a loop, fold the running offset into `o` exactly
      as Rule 2 does for descriptors. Report this under "Changes made" (it emits
      no descriptor), and flag it off the 2D-block fast path.

12. **Block pointer received as a helper argument.** A block pointer can never
    arrive from host launch (there is no host-side block-pointer object); it can
    only be a `@triton.jit` → `@triton.jit` parameter — and arriving that way
    already defeats the fast path (a descriptor passed across `tt.call` loses its
    identity). If the kernel body uses a block pointer that was **not** created
    locally, the translation is **not body-only**:
    - Change the helper's signature to take raw `base, shape, strides, offsets`
      (and `block_shape`/`order` if not already `tl.constexpr` there).
    - Build the descriptor **inside** the helper with `tl.make_tensor_descriptor`.
    - Update **every call site** to pass the raw ingredients.
    Report this as a **caller-affecting change** and list the touched call sites.
    No-op for the common create-and-use-in-the-same-kernel case.

## Output format

1. **The fully translated `@triton.jit` kernel** — every block-pointer construct
   replaced. Preserve surrounding code, comments, and signatures (except Rule 12,
   which legitimately changes a helper signature).
2. **"Changes made"** — a short list: counts and each transformation applied,
   e.g. "Replaced 3 `make_block_ptr` with `make_tensor_descriptor`; removed 2
   `tl.advance`, introduced `off_k`; B loaded transposed via `.T`; flagged
   `stride_ak: tl.int64`". State plainly that **no host-launch changes are
   required on XPU**.
3. **Performance note (per descriptor)** — for each descriptor, say whether it is
   on the efficient path (2D, last-stride-1, locally-created, dot-feeding,
   OWord-pitched → same 2D-block-I/O the block pointer used). For any descriptor
   that could **not** be put on the fast path, flag it with the reason
   (untraceable across `tt.call`, non-OWord pitch, not feeding a `tl.dot`,
   rank>2 unfoldable, rank-1 unit-stride 1D descriptor). A non-unit-stride rank-1
   block pointer (Rule 11) emits **no** descriptor — report it under "Changes
   made" as a masked tensor-of-pointer load on the pointer path, not as a
   per-descriptor note. Do not over-promise: the README's ">2x" figure is
   vs *tensor-of-pointers*, not vs block pointers.
4. **Caller-affecting changes** (only if Rule 12 fired) — the changed helper
   signature and the list of call sites to update.

## Self-check before returning

- Zero `tl.make_block_ptr` / `tl.advance` remain (unless Rule 12 deliberately
  keeps a block-ptr arg pending a caller fix you've flagged).
- Number of `make_tensor_descriptor` ≈ number of distinct block pointers, minus
  any non-unit-stride rank-1 pointers that became masked tensor-of-pointer loads
  (Rule 11).
- Every descriptor has last stride == 1 (transposes via `.T`, not last-stride≠1).
- **No descriptor was emitted for a non-unit-stride rank-1 block pointer** — each
  such pointer became a masked `tl.load`/`tl.store` whose `mask=` reproduces the
  original `boundary_check`, flagged off the fast path (Rule 11). (A `strides=(S,)`
  descriptor with `S != 1` would not compile.)
- Descriptors are created in the function that loads from them (Rule 12).
- `padding_option="nan"` carried to the constructor iff the source used it.
- No `triton.set_allocator` / host-launch change anywhere in the output.
