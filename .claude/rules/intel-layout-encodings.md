---
description: 'Intel GPU layout encodings: DPAS, Warp, Subgroup2DBlock, DotOperand, shared memory — shape derivation, layout conversion, encoding selection by passes'
applyTo: '**/Dialect/TritonIntelGPU/**/*.td, **/Dialect/TritonIntelGPU/**/*.cpp, **/Dialect/TritonIntelGPU/**/*.h, **/TritonIntelGPUTransforms/**/*.cpp, **/TritonIntelGPUTransforms/**/*.h, **/TritonIntelGPUToLLVM/**/*.cpp, **/Analysis/**/*.cpp, **/Analysis/**/*.h, **/Analysis/**/*.tpp'
---

# Intel GPU Layout Encodings

## Encoding Hierarchy

The Intel backend extends Triton's DistributedEncoding framework with three GPU-specific encoding attributes. All participate in the 4-level compute hierarchy: CTAs Per CGA -> Warps Per CTA -> Threads Per Warp -> Values Per Thread.

```
LayoutEncodingTrait (interface: getCGALayout(), getRank())
  |
DistributedEncodingTrait (interface: getRepOrder(), getTotalElemsPerThread(), toLinearLayout())
  |
DistributedEncoding (base class, combines DistributedEncodingTrait + LayoutEncodingTrait)
  |
  +-- BlockedEncodingAttr        (upstream, memory-coalesced access)
  +-- DpasEncodingAttr           (Intel, #ttig.dpas, + MmaEncodingTrait)
  +-- WarpEncodingAttr           (Intel, #ttig.warp)
  +-- Subgroup2DBlockEncodingAttr (Intel, #ttig.subgroup_2d_block, + MmaEncodingTrait)
  +-- LinearEncodingAttr         (upstream, flexible coordinate mapping)

MmaEncodingTrait (interface: getRepOrderForOperand(int opIdx))
  +-- DpasEncodingAttr
  +-- Subgroup2DBlockEncodingAttr

DotOperandEncodingAttr  (upstream wrapper, references parent MMA encoding)
  +-- parent = DpasEncodingAttr (Intel) or NvidiaMmaEncoding, AMDMfmaEncoding, etc.

SwizzledSharedEncodingAttr  (upstream, bank-conflict-aware shared memory layout)
  +-- constructed from DotOperandEncodingAttr via composeSharedLayoutForOperand()
```

## DpasEncodingAttr (`#ttig.dpas`)

The primary compute encoding for Intel XMX matrix operations.

### Parameters

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `repeatCount` | unsigned | Range [1, 8] | M dimension of DPAS tile |
| `systolicDepth` | unsigned | Must be **8** | Depth of systolic array (hardware-fixed) |
| `executionSize` | unsigned | **16** (PVC/BMG) or **8** (DG2) | N dimension (SIMD width) |
| `opsPerChannel` | unsigned | 1, 2, or 4 | K packing factor: `32 / element_bitwidth` |
| `warpsPerCTA` | ArrayRef<unsigned> | Rank 2 or 3 | Warp distribution in CTA, e.g., [4, 1] |
| `repCluster` | ArrayRef<unsigned> | Rank 2 or 3 | Repetition cluster size for memory optimization |
| `threadsPerWarp` | unsigned | Must be **16**, must be >= executionSize | Subgroup size for DPAS |
| `fp4KPack` | optional<unsigned> | If present, must be **2**; only valid when opsPerChannel=4 | FP4 packing along K |

### opsPerChannel by Element Type

| Element Type | Bitwidth | opsPerChannel | K (depth=8) |
|-------------|----------|---------------|-------------|
| TF32 | 32 | 1 | 8 |
| BF16, FP16 | 16 | 2 | 16 |
| INT8, FP8 (E5M2/E4M3FN) | 8 | 4 | 32 |
| FP4 (E2M1) | 4 | 4 (with fp4KPack=2) | 64 |

### Instruction Shape Derivation

Single DPAS instruction tile:
```
instShape_A = [repeatCount, systolicDepth * opsPerChannel]
instShape_B = [systolicDepth * opsPerChannel, executionSize]
instShape_C = [repeatCount, executionSize]
```

With repCluster scaling (per-warp tile):
```
shapeA = [repeatCount * repCluster[M], systolicDepth * opsPerChannel]
shapeB = [systolicDepth * opsPerChannel, executionSize * repCluster[N]]
shapeC = [repeatCount * repCluster[M], executionSize * repCluster[N]]
```

### Repetition Calculation

For a tensor of given `shape`, the number of DPAS tile repetitions per warp:

**Operand A** (opIdx=0):
```
rep[batch] = shape[0] / (shapeA[0] * warpsPerCTA[0])    // if rank 3
rep[M]     = shape[M] / (shapeA[M] * warpsPerCTA[M])
rep[K]     = shape[K] / shapeA[K]                         // K shared across warps
```

**Operand B** (opIdx=1):
```
rep[batch] = shape[0] / (shapeB[0] * warpsPerCTA[0])    // if rank 3
rep[K]     = shape[K] / shapeB[K]                         // K shared across warps
rep[N]     = shape[N] / (shapeB[N] * warpsPerCTA[N])
```

**Operand C/Result** (opIdx=2):
```
rep[batch] = shape[0] / (shapeC[0] * warpsPerCTA[0])    // if rank 3
rep[M]     = shape[M] / (shapeC[M] * warpsPerCTA[M])
rep[N]     = shape[N] / (shapeC[N] * warpsPerCTA[N])
```

### Elements Per Thread

```
elemsPerThread = (product(tileShape) / threadsPerWarp) * product(repetitions)
```

DPAS operand scalars are evenly sharded: each work-item gets `totalTileElements / threadsPerWarp` scalars per tile, multiplied by the number of tile repetitions.

### LinearLayout Basis Vectors

DPAS uses register and lane bases to map thread/register coordinates to tensor coordinates:

**Operand A**: Lane bases distribute threads across K columns first (one scalar per lane), then wrap to rows. Register bases pack elements vertically (for sub-32-bit types, multiple ops per lane). K dimension is broadcast across warps (zeros in warp basis).

**Operand B**: Lane bases distribute threads across N columns first, then wrap to K rows. Register bases pack K elements. K dimension is broadcast across warps.

**Operand C**: Lane bases distribute threads across N columns, then wrap to M rows. Register bases extend the M dimension. Both M and N are distributed across warps.

### repCluster Constraints

`repCluster` is computed by `calculateRepCluster()` with hardware 2D block I/O limits:
- `repCluster[M]` limited by max 2D block load tile_height (32 rows)
- `repCluster[N]` limited by max 2D block load bytes per row (64 bytes)
- Both values are powers of 2

## WarpEncodingAttr (`#ttig.warp`)

Thread-tile distribution encoding for non-DPAS operations.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sizePerThread` | ArrayRef<unsigned> | Elements computed per thread in each dimension |
| `threadsPerWarp` | ArrayRef<unsigned> | Number of threads per warp in each dimension |
| `order` | ArrayRef<unsigned> | Access order (fastest-changing dimension first) |

### Element Distribution
```
elemsPerThread[i] = sizePerThread[i] * threadsPerWarp[i]
```

Each thread owns a contiguous tile of `sizePerThread` elements, and `threadsPerWarp` threads are distributed across each dimension.

## Subgroup2DBlockEncodingAttr (`#ttig.subgroup_2d_block`)

Encoding for 2D block I/O layouts using Intel hardware 2D block load/store instructions.

### Parameters

| Parameter | Type | Constraint | Description |
|-----------|------|------------|-------------|
| `warpsPerCTA` | ArrayRef<unsigned> | Rank 2 | Warp distribution in CTA |
| `CGALayout` | CGAEncodingAttr | — | CTA layout encoding |
| `instrShape` | ArrayRef<unsigned> | Rank 2 (height, width) | Shape of 2D block operation |
| `numBlocks` | unsigned | — | Count of vertically adjacent blocks per load |
| `order` | ArrayRef<unsigned> | Rank 2 | Access order |
| `kWidth` | unsigned | 1, 2, or 4 | Layout conversion parameter for K dimension |
| `threadsPerWarp` | unsigned | Must be **16** | Subgroup size for 2D block I/O |

### Layout Mapping

```
packedElementsPerLane = ceil(instrShape[width] / threadsPerWarp)
rowsPerWarp = ceil(threadsPerWarp / lanes_used_for_columns)
```

- If `width == threadsPerWarp`: one column per lane, row-major mapping
- If `width < threadsPerWarp`: multiple rows per register to fill warp
- If `width > threadsPerWarp`: multiple elements per register (packed)

Increasing `numBlocks` scales the inner (width) dimension.

### Instruction Shape Selection

`getInstrShapeForLayout()` selects 2D block instruction parameters based on:
- Memory access pattern (row-major vs column-major)
- Element size in bits
- kWidth parameter
- Hardware limits from 2D block I/O constraints

## DotOperandEncodingAttr (Upstream Wrapper)

Wraps a parent MMA encoding (e.g., DpasEncodingAttr) with operand-specific metadata.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `opIdx` | unsigned | — | 0 = Operand A, 1 = Operand B |
| `parent` | Attribute | — | Parent MMA encoding (DpasEncodingAttr for Intel) |
| `kWidth` | unsigned | 0 | K-dimension element packing width |

### kWidth for DPAS

The `kWidth` parameter is tied to DPAS `opsPerChannel`:
- **Operand A**: `kWidth = opsPerChannel / 2` (when opsPerChannel >= 2) or `opsPerChannel` (when == 1)
- **Operand B**: `kWidth = opsPerChannel`

Validation ensures consistency between `DotOperandEncodingAttr.kWidth` and the parent `DpasEncodingAttr.opsPerChannel`.

## Shared Memory Layout

### SwizzledSharedEncodingAttr

When converting between DPAS and other distributed layouts, intermediate shared memory (SLM) is used with swizzled encoding to avoid bank conflicts.

Construction from DotOperandEncoding:
```
SwizzledSharedEncodingAttr::get(ctx, dotOpEnc, shape, sharedOrder, CGALayout, elementType)
```

The builder extracts `dotOpEnc.getKWidth()` and calls the backend-specific `composeSharedLayoutForOperand()` method to compute `vec`, `perPhase`, and `maxPhase` swizzle parameters.

### Bank Conflict Avoidance via Padding

For sub-group transpose operations through SLM, padding is added to prevent bank conflicts:
```
rowLength = threadsPerWarp + 1    // Extra element per row avoids bank conflicts
numMatrixCells = (numElements / subGroupSize) * (subGroupSize + 1)
scratchBytes = numMatrixCells * bytesPerElement
```

The `+1` padding ensures that adjacent rows map to different SLM banks.

### Shared Memory Order

`sharedOrder` is computed to move the batch dimension to the end, minimizing bank conflicts for the common 2D access patterns.

## Layout Conversion Flow

### When Conversions Are Inserted

Layout conversions (`ConvertLayoutOp`) are inserted by these passes in order:

1. **Coalesce** — Inserts conversions to BlockedEncodingAttr for memory-coalesced access
2. **AccelerateMatmul** — Inserts conversions to DpasEncodingAttr/DotOperandEncodingAttr around dot operations
3. **MaterializeBlockPointer** — Sets block I/O attributes (row_major/column_major) on operations
4. **OptimizeDotOperands** — Propagates DotOperandEncodingAttr closer to loads

### When Conversions Are Removed

**RemoveLayoutConversions** operates in phases:

1. **Anchor layout initialization** — Mark operations with fixed layouts (expensive load/store, dot, etc.)
2. **Forward propagation** — Propagate layouts from anchors until fixpoint using `inferDstEncoding()`
3. **Conflict resolution** — Prefers BlockedEncodingAttr when layouts conflict
4. **Backward rematerialization** — `LayoutRematerialization` class hoists ConvertLayoutOps:
   - `hoistConvertDotOperand()` — before dot operand consumers
   - `hoistConvertOnTopOfExtOrBroadcast()` — before extension/broadcast ops
   - `hoistConvertIntoConditionals()` — into conditional branches
5. **Canonicalization** — Removes redundant identity conversions

### Layout Propagation Utilities

**`inferSrcEncoding(op, encoding)`** — Infers source encoding from an operation and target encoding:
- For `MakeTensorPtrOp` and `AdvanceOp`: returns encoding unchanged
- For `DotEncoding + DPASEncoding`: dispatches to `DialectInferLayoutInterface` (e.g., for `Fp4ToFpOp`)
- Falls back to upstream `mlir::inferSrcEncoding()`

**`getConvertBackwardSlice(root, targetEncoding, ...)`** — Core backward slice algorithm:
1. Enqueue root operand with target encoding
2. Process queue: pop (operand, encoding), propagate through defining ops using `inferSrcEncoding`
3. Handle special cases: `scf::ForOp` (init + yield), `scf::IfOp` (both branches), block arguments
4. Track seen pairs to avoid cycles
5. Fail on `CatOp` (cannot propagate); special handling for `GatherOp` (only through indices)

## Encoding Selection by Pass

### Coalesce — BlockedEncodingAttr Selection

Selects memory-coalesced layouts using `ModuleAxisInfoAnalysis`:

1. Extract pointer access axis info (contiguity, divisibility)
2. Determine memory access order from contiguity analysis
3. Find multi-root slice: operations with same shape and order
4. Calculate elements per thread via `getNumElementsPerThread()`
5. Enforce max **128 bits per thread** for store operations
6. Construct `BlockedEncodingAttr::get(ctx, shape, sizePerThread, order, numWarps, threadsPerWarp, CGALayout)`

### AccelerateMatmul — DpasEncodingAttr Construction

Converts blocked encodings to DPAS for matrix multiply:

1. Check DPAS applicability: `DPASAnalysisFactory::canUseDPAS()`
2. Query capability: `DPASCapability::getDPASCapability(mod)`
3. Determine `opsPerChannel` from element bitwidth (32/bitwidth)
4. Clamp `repeatCount` to min(capability.repeatCount, M-dimension)
5. Compute `warpsPerTile` via `calculateWarpsPerTile()`
6. Compute `repCluster` via `calculateRepCluster()` respecting 2D block I/O limits
7. Construct: `DpasEncodingAttr::get(ctx, repeatCount, systolicDepth, executionSize, opsPerChan, warpsPerTile, repCluster, threadsPerWarp, fp4Flag)`
8. Wrap operands: `DotOperandEncodingAttr::get(ctx, opIdx, dpasEnc, kWidth)`
9. Insert `ConvertLayoutOp` before dot (for A, B, C) and after dot (back to original)
10. For `DotScaledOp`: construct `LinearEncodingAttr` from `BlockScaledDPAStoLinearLayout()` for scale operands

### MaterializeBlockPointer — Block I/O Attribute Setting

Sets `ttig.block_io` string attributes, not encoding attributes:

**For tensor block pointers:**
1. Validate 2D block read alignment
2. Determine row-major vs column-major from stride-one dimension
3. Check pitch alignment (128-bit aligned)
4. Inspect dot operand layout to avoid transpose performance regression
5. Set: `op->setAttr("ttig.block_io", StringAttr::get(ctx, "row_major"/"column_major"))`

**For tensors of pointers:**
1. Analyze contiguity and stride via axis info
2. Validate rows match tensor dimension, stride is 16-byte multiple, base pointer is 4-byte aligned
3. Apply both row_major and column_major checks independently

### ReduceDataDuplication — Shared Memory Intermediate

Eliminates redundant register copies by routing through shared memory:

1. Replace direct ConvertLayoutOp with two-stage: `LocalAlloc` -> `LocalLoad`
2. Construct `SwizzledSharedEncodingAttr::get(ctx, dstDotOp, shape, sharedOrder, CGALayout, elementType)`
3. `sharedOrder` reorders dimensions (batch dimension moved to end)

### OptimizeDotOperands — Encoding Propagation

Pushes DotOperandEncodingAttr closer to loads by converting the operand early, reducing register pressure from layout conversions near the dot operation.

## Layout Conversion Lowering

### ConvertLayoutOp to LLVM

Three lowering patterns (in order of priority):

1. **LinearLayout-based conversion** (benefit +2): Highest priority
   - Detects sub-group shuffles: `cvtIsSubGroupShuffle(srcTy, dstTy)` — lane reordering via shuffles
   - Detects sub-group transposes: `cvtIsSubGroupTranspose(srcTy, dstTy)` — transpose via shared memory
   - Returns failure for other cases (falls through to lower-priority patterns)

2. **Guard pattern** (benefit +1): Asserts Intel-specific cases (shuffle/transpose) were handled; returns failure for all other conversions

3. **Upstream patterns** (base benefit): Generic `ConvertLayoutOp` lowering from upstream Triton

### Sub-Group Shuffle Detection

Uses LinearLayout composition: `dstLayout.invertAndCompose(srcLayout)`. Valid shuffle if:
- Register-to-register basis vectors form identity
- Lane-to-register basis vectors are zero
- Register-to-lane basis vectors describe the shuffle pattern

### Sub-Group Transpose via SLM

When source and destination layouts differ by a transpose:

1. **Store phase**: Each thread stores its elements to SLM at `row * rowLength + col`
   - `rowLength = threadsPerWarp + 1` (padding for bank conflict avoidance)
   - Uses `SubGroupBlockWriteOp` for efficient writes
2. **Load phase**: Each thread loads back in the transposed order using vectorized loads
3. Supports both 8/16/32/64-bit integer elements and bitcastable float types

