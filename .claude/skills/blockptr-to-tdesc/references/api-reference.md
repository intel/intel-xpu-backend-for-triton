# API Reference — Block Pointer vs Tensor Descriptor

Exact signatures and constraints, verified against Triton source
(`python/triton/language/core.py`, `python/triton/tools/tensor_descriptor.py`).
Line numbers are from the verification snapshot and may drift; the semantics are
what matter.

## Deprecated: block-pointer API

### `tl.make_block_ptr(base, shape, strides, offsets, block_shape, order)`
Returns a block pointer into a parent tensor. **Deprecated** — the implementation
emits `warn("tl.make_block_ptr is deprecated. Use TensorDescriptor or
tl.make_tensor_descriptor instead.")`.

| Param | Meaning |
|-------|---------|
| `base` | scalar pointer to the parent tensor |
| `shape` | parent tensor shape (per dim), used for boundary masking |
| `strides` | parent tensor strides (per dim), in elements |
| `offsets` | starting offsets of the block (per dim) |
| `block_shape` | tile shape to load/store (constexpr per dim) |
| `order` | physical-memory permutation. `(1, 0)` = row-major / last dim contiguous; `(0, 1)` = column-major / transposed read |

- Rank: **≥ 1**, no upper bound (only `rank == 0` is rejected). Tests exercise
  rank-1 and rank-3 block pointers.

### `tl.advance(block_ptr, offsets)`
Returns a **new** block pointer with `offsets` added to the current offsets
(per-dim **delta**, not absolute). Must assign the result — it has no side
effect. Only valid on block pointers created by `make_block_ptr`.

### `tl.load(block_ptr, boundary_check=(), padding_option="")`
- `mask` and `other` **must be `None`** for block pointers — the load raises
  `ValueError` otherwise.
- `boundary_check`: tuple of dim indices to bounds-check.
- `padding_option`: `""` (undefined), `"zero"`, or `"nan"` (float only) — value
  used for out-of-bounds elements.

### `tl.store(block_ptr, value, boundary_check=())`
- No `mask`; out-of-bounds writes are ignored via `boundary_check`.
- No `padding_option` (irrelevant for stores).

## Modern: device-side tensor-descriptor API

### `tl.make_tensor_descriptor(base, shape, strides, block_shape, padding_option="zero")`
Returns a `tensor_descriptor`. Created **inside** a `@triton.jit` kernel
(device-side).

| Param | Meaning |
|-------|---------|
| `base` | scalar pointer; must be **16-byte aligned** |
| `shape` | tensor shape (per dim); also masks OOB on load/store |
| `strides` | strides (per dim). **Last stride must be 1** (contiguous); leading strides must be multiples of 16 bytes |
| `block_shape` | tile shape (constexpr per dim) |
| `padding_option` | `"zero"` (default, any dtype) or `"nan"` (float only) |

- Rank: code allows **1–5** (`rank > 0`, `rank <= 5`); docstring says "2-5".
- There is **no `offsets` and no `order`** — offsets are passed at load/store
  time, and orientation is handled by `.T` on the loaded block (see Rule 6).

### `desc.load([offsets])`
Loads a `block_shape` tile starting at the given **element** offsets (one per
dim). "Values outside of the tensor bounds will be filled with zeros." Offsets
should be a multiple of 16 bytes.

### `desc.store([offsets], value)`
Stores `value` (shape == `block_shape`) at the given element offsets. "Values
outside of the tensor bounds will be ignored."

## Constraint quick-reference (must hold to compile / be correct)

- `base` 16-byte aligned; **last stride == 1**; leading strides multiples of 16
  bytes.
- Descriptor block element type **==** the loaded/stored tensor element type, and
  total element counts match (`DescriptorLoadOp`/`StoreOp` verifier).
- `padding`: `"zero"` any dtype, `"nan"` float only.
- Rank 1–5 (XPU optimizes only 2D — see `performance-notes.md`).

## Host-side `TensorDescriptor` (do NOT use for XPU migration)

`triton.tools.tensor_descriptor.TensorDescriptor` is a host dataclass
(`from_tensor(...)`) passed to kernels — the NVIDIA/TMA style. The Intel XPU
guidance is **device-side** descriptors only. This skill always produces
`tl.make_tensor_descriptor` *inside* the kernel; it never emits a host-side
`TensorDescriptor` or a `triton.set_allocator` call.
