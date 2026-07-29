# TMA Descriptor Update — Diagnostic Notes

Investigation into why `tl.make_tensor_descriptor` underperforms when used inside
a loop on Hopper+, and what update primitive could fix it.

## Background

`tl.make_tensor_descriptor` is the unified descriptor API. The compiler picks
its lowering by hardware:

| Backend | Lowering | When |
|---|---|---|
| Nvidia (SM ≥ 9.0) | `add_tma_lowering` → `ttng.tensormap_create` + fenceproxy | Hopper / Blackwell |
| Nvidia (SM < 9.0) | `add_rewrite_tensor_descriptor_to_pointer` → plain `tt.load`/`tt.store` with computed pointer + mask | Pre-Hopper |
| AMD | `add_rewrite_tensor_descriptor_to_pointer` (always) | All |
| Intel | `add_rewrite_tensor_descriptor_to_pointer` (always) | All |

The pivot for Nvidia is in [third_party/nvidia/backend/compiler.py:247-248,315-316](../third_party/nvidia/backend/compiler.py#L247-L248):

```python
if capability // 10 < 9:
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
...
if capability // 10 >= 9:
    nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
```

Once Nvidia takes the TMA path, every `tt.make_tensor_descriptor` becomes a
real on-device descriptor. There is no fallback to tensor-of-pointers on
Hopper+.

## The Problem

When `make_tensor_descriptor` lives inside a loop and the descriptor's `base`
depends on a per-iteration value, every iteration emits a full
`ttng.tensormap_create` — even if the only field that actually changes is
`global_address`.

`TMACreateDescLowering` ([lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp:165-184](../lib/Dialect/TritonNvidiaGPU/Transforms/TMALowering.cpp#L165-L184))
unconditionally rewrites each `MakeTensorDescOp` to:
1. `triton_gpu.global_scratch_alloc` (128 B per descriptor)
2. `ttng.tensormap_create` (writes 16 fields)
3. `ttng.tensormap_fenceproxy_acquire`

Each `tensormap_create` then expands at LLVM lowering
([third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp:243-297](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp#L243-L297))
into ~16 PTX `tensormap.replace.tile.<field>.shared::cta.b1024.<bN>`
instructions — `global_address`, `rank`, `box_dim[i]`, `global_dim[i]`,
`global_stride[i]`, `element_stride[i]`, `elemtype`, `interleave_layout`,
`swizzle_mode`, `fill_mode` — followed by `tensormap.cp_fenceproxy` to global
memory and a fenceproxy.acquire round-trip.

The pipeliner ([lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp:595-671](../lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp#L595-L671))
multi-buffers the alloc (one slot per pipeline stage, ring-counter-indexed) so
the writes can overlap with prior loads, but it still issues the full
`createTMADesc` writeout each trip. Comment at
[PipeliningUtility.cpp:695-698](../lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp#L695-L698)
explicitly admits this is "duplicate of existing tma descriptors."

## Concrete Example (vLLM Unified Attention)

The failing pattern is in
[benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch:80-90](../benchmarks/triton_kernels_benchmark/vllm/unified_attention/unified_attention.patch#L80-L90):

```python
for j in ...:
    physical_block_idx = tl.load(block_tables_ptr + block_table_offset
                                 + j // (BLOCK_SIZE // TILE_SIZE)).to(tl.int64)

    v_base = value_cache_ptr + physical_block_idx * stride_v_cache_0 + kv_head_idx * stride_v_cache_2
    v_desc = tl.make_tensor_descriptor(
        base=v_base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_v_cache_1, stride_v_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))

    k_base = key_cache_ptr + physical_block_idx * stride_k_cache_0 + kv_head_idx * stride_k_cache_2
    k_desc = tl.make_tensor_descriptor(
        base=k_base,
        shape=(BLOCK_SIZE, HEAD_SIZE),
        strides=(stride_k_cache_1, stride_k_cache_3),
        block_shape=(TILE_SIZE, HEAD_SIZE_PADDED))

    K_load = k_desc.load([offset_in_block, 0]).T
    V_load = v_desc.load([offset_in_block, 0])
```

Only `v_base` / `k_base` change between iterations (paged-KV indirection through
`block_tables`). `shape`, `strides`, `block_shape` are loop invariants. The
compiler can't hoist the descriptor build because `physical_block_idx` is itself
a per-iteration `tl.load`.

## Available Update Primitive

PTX exposes per-field replace operations. Triton's Nvidia backend already uses
the C++ helpers internally, but they're not surfaced as IR ops.
From [TMAToLLVM.cpp:46-182](../third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TMAToLLVM.cpp#L46-L182):

| Helper | PTX | What changes |
|---|---|---|
| `tensormap_replace_global_address` | `tensormap.replace.tile.global_address.shared::cta.b1024.b64` | base pointer |
| `tensormap_replace_rank` | `…rank.b1024.b32` | rank |
| `tensormap_replace_box_dim` | `…box_dim.b1024.b32` (per-ord) | per-dim tile size |
| `tensormap_replace_global_dim` | `…global_dim.b1024.b32` (per-ord) | per-dim global shape |
| `tensormap_replace_global_stride` | `…global_stride.b1024.b64` (per-ord) | per-dim stride |
| `tensormap_replace_element_stride` | `…element_stride.b1024.b32` (per-ord) | element stride |
| `tensormap_replace_elemtype` | `…elemtype.b1024.b32` | dtype enum |
| `tensormap_replace_interleave_layout` | `…interleave_layout.b1024.b32` | interleave |
| `tensormap_replace_swizzle_mode` | `…swizzle_mode.b1024.b32` | swizzle |
| `tensormap_replace_fill_mode` | `…fill_mode.b1024.b32` | fill mode |

For the unified-attention loop, only `global_address` changes. A single
`tensormap.replace.tile.global_address` followed by a fresh
`tensormap_fenceproxy_acquire` would replace ~15 redundant per-iteration writes.

What's missing in the IR: there is **no** `ttng.tensormap_replace` op today.
`tt.make_tensor_descriptor` ([include/triton/Dialect/Triton/IR/TritonOps.td:985](../include/triton/Dialect/Triton/IR/TritonOps.td#L985))
and `ttng.tensormap_create` ([include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td:1031](../include/triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td#L1031))
are the only descriptor-producing ops.

## Design Space for the Fix

Two shapes:

### 1. Explicit IR/language update op (smaller blast radius)

Add a `tt.tensor_descriptor_replace` op (or per-field variants) at the TT
dialect level, plumb it through TTGIR, and lower to
`tensormap.replace.tile.<field>` + fenceproxy on Nvidia (helpers already
exist), and to plain pointer-arith on Intel/AMD/pre-Hopper. Authors then write
`make_tensor_descriptor` at the top of the kernel and `desc.replace(base=…)` in
the loop. The vLLM patch becomes a one-line swap. Lowering is mechanical;
user code changes are explicit.

### 2. Compiler-driven LICM (no API change)

A pass that walks each `scf.for`, finds `MakeTensorDescOp`s where only some
fields depend on the IV, hoists the create above the loop, and synthesizes
per-field `tensormap_replace` ops in the loop body. Existing kernels "just
work." Risk: more analysis, easy to miss patterns (e.g., `tt.addptr` chains
feeding `base`), and harder to keep the pipeliner happy because the create
stops being a per-iteration anchor.

**Recommendation:** do (1) first — it's the load-bearing piece for (2) anyway,
and it lets us land the unified-attention win immediately by patching the
kernel. Then (2) becomes "synthesize the new op when possible" instead of
"build the whole machinery."

## Work Breakdown

1. TT op definition (`tt.tensor_descriptor_replace` or per-field).
2. TTGIR / Nvidia lowering → `ttng.tensormap_replace_*` + fenceproxy.
3. Intel rewrite-to-pointer handling (replace ≡ pointer arith on the cached descriptor tuple).
4. Pipeliner integration — make sure the existing multi-buffer counter scheme at [PipeliningUtility.cpp:623-671](../lib/Dialect/TritonGPU/Transforms/Pipeliner/PipeliningUtility.cpp#L623-L671) still applies (replace must run on the per-iteration buffer slot, not the original).
5. Kernel patch + bench against the current `tensormap_create`-per-iteration baseline.

## Reproducer

See [paged_kv_load.py](paged_kv_load.py) — minimal kernel that exhibits the
per-iteration `tensormap_create` cost.

### Why this kernel cannot be hoisted

The descriptor's `base` is computed as:

```python
physical_block_idx = tl.load(block_tables_ptr + bt_off + j).to(tl.int64)
kv_base = kv_cache_ptr + physical_block_idx * stride_kv_0 + KV_HEAD_IDX * stride_kv_2
```

- `j` is the loop induction variable, so `block_tables_ptr + bt_off + j` is
  loop-varying.
- `physical_block_idx` comes from `tl.load` of a runtime tensor — its value is
  unknown to the compiler and changes per iteration.
- `triton_licm` cannot hoist a `tl.load` that depends on the IV, and cannot
  fold `physical_block_idx * stride_kv_0` because `physical_block_idx` is a
  runtime SSA value, not a constant.
- The paged-KV scheme is genuinely non-contiguous: `block_tables` can map
  logical block `j` to any physical block in any order, so the access pattern
  cannot be re-expressed as a single hoisted descriptor with computed offsets
  (e.g. `desc.load([physical_block_idx * BLOCK_SIZE, 0])` only works if the
  KV cache is laid out as one big contiguous tensor, which it isn't).

This means the descriptor build *must* live inside the loop. The optimization
target is the per-iteration cost when the build cannot move, not LICM.

### Inspecting the IR

Run with `MLIR_ENABLE_DUMP=1` and look at the TTGIR after `add_tma_lowering`.
Inside the `scf.for` body, every iteration should contain:

```mlir
%alloc = ttg.global_scratch_alloc {alignment = 64, nbytes = 128} : !tt.ptr<i8>
ttng.tensormap_create %alloc, %kv_base,
    [%c16, %c128],          // box_dim       (loop-invariant)
    [%c16, %c128],          // global_dim    (loop-invariant)
    [%stride_kv_1_b],       // global_stride (loop-invariant)
    [%c1, %c1] {            // element_stride (loop-invariant)
      elem_type = ..., interleave_layout = 0, swizzle_mode = ..., fill_mode = 0
    }
ttng.tensormap_fenceproxy_acquire %alloc
%desc = ttng.reinterpret_tensor_descriptor %alloc
... use %desc in async TMA copy ...
```

Only `%kv_base` is loop-varying. The other 15-ish operands are dead weight.

## Implementation (what landed)

The fix routes loop-recreated descriptors to the existing pointer fallback
rather than adding a new update primitive. It is a `loop-recreated-only` mode on
`RewriteTensorDescriptorToPointer` (upstream Triton), gated in the NVIDIA
`make_ttir`:

- **Pre-Hopper** (`capability // 10 < 9`) or `TRITON_INTEL_NVIDIA_FORCE_TD_TEST`:
  run the pass in its original demote-**everything** mode (`loop-recreated-only=false`).
- **Hopper+**: run `loop-recreated-only=true` — demote only descriptors that are
  recreated in a loop *and* not hoistable; keep hoistable/out-of-loop descriptors
  on the TMA path.
- `TRITON_INTEL_NVIDIA_DISABLE_LOOP_TD_REWRITE`: skip the pass (raw
  `tensormap_create`, the un-optimized A/B baseline).

"Not hoistable" mirrors MLIR's LICM exactly: a descriptor is demoted iff some
operand stays loop-varying after LICM — checked recursively through pure ops
(`isLoopInvariantAfterLICM`), because this pass runs *before* `triton-licm`. This
is what keeps the paged-KV kernel demoted (its base traces to a per-iteration
`tt.load`, which is impure, so it stays loop-varying) while a descriptor built
from an in-loop temporary of invariant operands is correctly kept.

Selectivity is per-descriptor, driven by one forward provenance analysis
(`computeDescProvenance` → `computeDemotions`) that propagates from every
`MakeTensorDescOp` to the descriptor values it reaches. Dynamic conversion
legality is then a lookup: an op is illegal iff every descriptor it touches is
one we demote. Because the unit of rewriting is an *operation* (the type
converter expands every `!tt.tensordesc` in a signature at once), descriptors
that co-occur on an op decide together — all demoted, or all kept. A group is
kept whenever any member reaches something this mode cannot rewrite: a
`tt.return`/`tt.call` operand, an op outside the conversion target, or a
descriptor whose provenance is unnameable (function argument, call result,
`ub.poison`). Covered by
`test/Triton/rewrite-tensor-descriptor-loop-recreated.mlir`.

> Empirically on Blackwell the pointer fallback was ~2.5x faster than
> per-iteration TMA and hoisting didn't help — hence "demote", not "update in
> place". See `design-hypothesis.md` for the create-vs-update analysis this
> superseded.
