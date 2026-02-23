---
description: 'MLIR pass writing patterns for Intel XPU backend: pass declaration, structural patterns, MLIR best practices, utilities, and complete pass inventory'
applyTo: '**/intel/lib/**/*.cpp, **/intel/lib/**/*.h, **/intel/include/**/Transforms/**/*.td, **/intel/include/**/Transforms/**/*.h, **/intel/include/TritonGENToLLVM/**/*.td, **/intel/include/TritonIntelGPUToLLVM/**/*.td, **/intel/include/GPUToTritonGEN/**/*.td, **/intel/include/TritonAnnotateModule/**/*.td, **/triton_xpu.cc'
---

# Intel XPU Backend — MLIR Pass Writing Patterns

## 1. Pass Declaration (TableGen)

### TableGen File Locations

| Scope | TableGen File | Pass Prefix |
|-------|--------------|-------------|
| TTIR transforms | `intel/include/Dialect/Triton/Transforms/Passes.td` | `triton-intel-<name>` |
| TTGIR transforms | `intel/include/Dialect/TritonIntelGPU/Transforms/Passes.td` | `tritonintelgpu-<name>` |
| GPU→TritonGEN | `intel/include/GPUToTritonGEN/Passes.td` | `convert-gpu-to-tritongen` |
| TritonGEN→LLVM | `intel/include/TritonGENToLLVM/Passes.td` | `convert-tritongen-to-llvm` |
| TritonGEN→SPIR-V | `intel/include/TritonGENToSPIRV/Passes.td` | `convert-tritongen-to-spirv` |
| IntelGPU→LLVM | `intel/include/TritonIntelGPUToLLVM/Passes.td` | `convert-triton-intel-gpu-to-llvm` |
| Module annotation | `intel/include/TritonAnnotateModule/Passes.td` | `triton-annotate-module` |

### Basic Pass Declaration
```tablegen
def TritonIntelGPUCoalesce
    : Pass<"tritonintelgpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";
  let description = [{...}];
  let dependentDialects = [
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::intel::TritonIntelGPUDialect"
  ];
}
```

### Pass with Options
```tablegen
def TritonIntelGPUPipeline : Pass<"tritonintelgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "Pipeline loops";
  let dependentDialects = [...];
  let options = [
    Option<"numStages", "num-stages", "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"useBarrier", "use-barrier", "bool", /*default*/"false",
           "generate barriers">,
  ];
}
```

### Rules
- **Always** specify `dependentDialects` for every dialect the pass may create ops in
- CLI name convention: kebab-case, prefixed by dialect scope
- Operation target is typically `"mlir::ModuleOp"` for Intel passes
- Option types: `"int32_t"`, `"bool"`, `"std::string"` with string defaults

## 2. Pass Implementation — Structural Patterns

### Pattern A: Manual IR Walking

For complex, analysis-driven transforms where operation visit order matters.

**Examples**: Coalesce, MaterializeBlockPointer, RemoveMasks, ReduceVariableLiveness

```cpp
// In the mlir::triton::gpu::intel namespace:
#define GEN_PASS_DEF_TRITONINTELGPUCOALESCE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

struct CoalescePass
    : public ttgi::impl::TritonIntelGPUCoalesceBase<CoalescePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // 1. Run analysis
    tt::intel::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    // 2. Walk IR collecting decisions
    llvm::MapVector<Operation *, Attribute> layoutMap;
    moduleOp.walk([&](Operation *curr) {
      // Analyze and populate layoutMap
    });

    // 3. Apply transformations
    for (auto &[op, layout] : layoutMap)
      coalesceOp(layout, op);
  }
};
```

**When to use**: Operation visit order matters, need analysis results before rewriting, complex multi-step transforms.

### Pattern B: Pattern-Based Rewrite

For independent, local rewrites where operations can be transformed individually.

**Examples**: AccelerateMatmul, OptimizeDotOperands, ReduceDataDuplication

```cpp
#define GEN_PASS_DEF_TRITONINTELGPUACCELERATEMATMUL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

// Define rewrite pattern
class BlockedToDPAS : public OpRewritePattern<tt::DotOp> {
public:
  BlockedToDPAS(MLIRContext *context, int benefit = 1)
      : OpRewritePattern<tt::DotOp>(context, benefit) {}

  LogicalResult matchAndRewrite(tt::DotOp op,
                                PatternRewriter &rewriter) const override {
    // Guard: return failure() if pattern doesn't apply
    if (isa<ttgi::DpasEncodingAttr>(op.getType().getEncoding()))
      return failure();

    // Perform rewrite using rewriter API exclusively
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, oldRetType, res);
    return success();
  }
};

// Define pass
struct AccelerateMatmulPass
    : public ttgi::impl::TritonIntelGPUAccelerateMatmulBase<
          AccelerateMatmulPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<BlockedToDPAS>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};
```

**When to use**: Independent local rewrites, canonicalization-style, no global ordering dependency.

### Pattern C: Conversion/Lowering

For cross-dialect lowering that requires type conversion.

**Examples**: TritonGENToLLVM, TritonIntelGPUToLLVM, GPUToTritonGEN

```cpp
#define GEN_PASS_DEF_CONVERTTRITONGENTOLLVM
#include "intel/include/TritonGENToLLVM/Passes.h.inc"

// Define conversion pattern
struct DPASLowering
    : public ConvertOpToLLVMPattern<TritonGEN::MatrixDPASOp> {
  using ConvertOpToLLVMPattern<
      TritonGEN::MatrixDPASOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TritonGEN::MatrixDPASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use adaptor for already-converted operands
    // Use rewriter for all mutations
    return success();
  }
};

// Define pass with TypeConverter and ConversionTarget
struct ConvertTritonGENToLLVM
    : public triton::impl::ConvertTritonGENToLLVMBase<ConvertTritonGENToLLVM> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);
    LLVMConversionTarget target(*ctx);
    RewritePatternSet patterns(ctx);

    populateTritonGENToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
```

**Multi-phase lowering** (TritonIntelGPUToLLVM — 3 phases):
1. Lower functions (function signatures, calling conventions)
2. Lower call/ret operations
3. Lower all remaining operations (bulk conversion)

Each phase uses a separate `RewritePatternSet` and `applyPartialConversion()` call.

**Pattern collection via populate functions**:
```cpp
void populateConversionPatterns(RewritePatternSet &patterns, ...) {
  intel::populateLoadStoreOpToLLVMPatterns(typeConverter, patterns, ...);
  intel::populateDotOpToLLVMPatterns(typeConverter, patterns, ...);
  intel::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ...);
  // ... 28+ populate functions
}
```

**When to use**: Cross-dialect lowering, type conversions needed, ConversionTarget legality required.

### Pattern D: Capability-Gated Pass

Guard hardware-specific transforms with module attribute checks.

**Examples**: MaterializeBlockPointer, Pipeline, AccelerateMatmul

```cpp
void runOnOperation() override {
  ModuleOp mod = getOperation();

  // Early exit if hardware doesn't support this feature
  if (!mod->hasAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
    return;

  // Proceed with transformation...
}
```

Often combined with Pattern A or B. Multiple attributes may be checked.

## 3. MLIR Best Practices

### Pattern Rewriting Rules

These rules are **mandatory** per official MLIR documentation:

1. **No premature mutation**: Never modify IR before the match is confirmed successful. All guard checks first, mutations after.
2. **Use PatternRewriter exclusively**: All IR creation, modification, and deletion must go through the `PatternRewriter` API. Never bypass it.
3. **Return semantics**: `matchAndRewrite` must return `success()` if and only if IR was modified. Return `failure()` otherwise.
4. **Root operation**: The matched root op must be updated in-place, replaced, or erased by the pattern.
5. **Diagnostic messages**: Use `rewriter.notifyMatchFailure(op, "reason")` for debug diagnostics.
6. **Recursion safety**: If a pattern can match its own output, call `setHasBoundedRewriteRecursion()` during initialization.

### Pass Management Rules

1. **No mutable state across invocations**: `runOnOperation()` must not depend on state from previous calls (required for thread safety).
2. **Declare dependent dialects**: List all dialects whose ops may be created in `dependentDialects` in the `.td` file.
3. **Signal failure**: Call `signalPassFailure()` when `applyPatternsGreedily()` or `applyPartialConversion()` fails. Never silently continue.
4. **Analysis preservation**: Call `markAllAnalysesPreserved()` when IR is not modified. Use `getAnalysis<T>()` for lazy cached queries.

### Driver Selection

| Driver | Function | Use Case |
|--------|----------|----------|
| Greedy | `applyPatternsGreedily()` | Transformation passes. Iterates to fixed-point. |
| Partial Conversion | `applyPartialConversion()` | Lowering passes. Requires `ConversionTarget`. |
| Walk | `walkAndApplyPatterns()` | Simple one-shot traversals. No revisiting. |

### Pattern Base Class Reference

| Base Class | Signature | TypeConverter | Use Case |
|-----------|-----------|---------------|----------|
| `OpRewritePattern<Op>` | `matchAndRewrite(op, PatternRewriter &)` | No | Intra-dialect transforms |
| `ConvertOpToLLVMPattern<Op>` | `matchAndRewrite(op, OpAdaptor, ConversionPatternRewriter &)` | Yes | Lowering to LLVM |
| `ConvertTritonGPUOpToLLVMPattern<Op>` | Same as above + target info | Yes | Triton GPU → LLVM |
| `OpConversionPattern<Op>` | `matchAndRewrite(op, OpAdaptor, ConversionPatternRewriter &)` | Yes | Generic dialect conversion |

## 4. Namespace and Macro Conventions

### GEN_PASS_DEF Placement

```cpp
// TTGIR transforms — in gpu::intel namespace
namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUCOALESCE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

// TTIR transforms — in triton namespace
namespace mlir::triton {
#define GEN_PASS_DEF_TRITONINTELREMOVEMASKS
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton
```

### Pass Header (Passes.h)

```cpp
namespace mlir::triton::gpu::intel {
#define GEN_PASS_DECL
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
#define GEN_PASS_REGISTRATION
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel
```

### Standard Aliases

```cpp
using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;
```

### Debug Macros

```cpp
#define DEBUG_TYPE "tritonintelgpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// Usage
LDBG("Processing op: " << *op);
LLVM_DEBUG({
  llvm::dbgs() << "[" DEBUG_TYPE "]: Detailed info\n";
  op->dump();
});
```

Debug filtering: `-debug-only=tritonintelgpu-coalesce` or `-debug-only=greedy-rewriter`

## 5. Python Binding (triton_xpu.cc)

### No-Option Pass
```cpp
ADD_PASS_WRAPPER_0("add_coalesce", gpu::intel::createTritonIntelGPUCoalesce);
```
Python: `intel.passes.ttgpuir.add_coalesce(pm)`

### Pass with Scalar Options
```cpp
ADD_PASS_OPTION_WRAPPER_2("add_pipeline",
                          gpu::intel::createTritonIntelGPUPipeline, int, bool);
```
Python: `intel.passes.ttgpuir.add_pipeline(pm, opt.num_stages, opt.use_barrier)`

### Pass with Struct Options
```cpp
py::class_<gpu::intel::TritonAnnotateModuleOptions>(m, "AnnotateModuleOptions")
    .def(py::init<>())
    .def_readwrite("min_sg_size",
                   &gpu::intel::TritonAnnotateModuleOptions::minSGSize)
    .def_readwrite("support_2d_block_io",
                   &gpu::intel::TritonAnnotateModuleOptions::support2DBlockIO)
    // ...
;
ADD_PASS_OPTION_WRAPPER_1("add_triton_annotate_module",
                          gpu::intel::createTritonAnnotateModule,
                          gpu::intel::TritonAnnotateModuleOptions);
```

### Python Pass Namespaces
- `intel.passes.ttir.add_<name>(pm)` — TTIR passes
- `intel.passes.ttgpuir.add_<name>(pm)` — TTGIR passes and lowering
- `intel.passes.arith.add_<name>(pm)` — Arith emulation passes

## 6. Common Utilities

### Layout and Encoding Queries (`Utility.h`)
```cpp
bool hasDpasEncoding(RankedTensorType t);        // Has DpasEncodingAttr
bool hasDotDpasEncoding(RankedTensorType t);     // Has DotOperand with DPAS parent
std::optional<DotOperandEncodingAttr> getDotEncoding(RankedTensorType t);
Attribute inferSrcEncoding(Operation *op, Attribute encoding);
```

### DPAS Calculation Helpers
```cpp
SmallVector<unsigned> calculateDPASInstShapeA(unsigned rc, unsigned sd, unsigned opc);
SmallVector<unsigned> calculateWarpsPerTile(unsigned capRC, unsigned capExec,
                                            ArrayRef<int64_t> shape, unsigned numWarps);
SmallVector<unsigned> calculateRepCluster(const DPASCapability &cap, ...);
```

### Layout Propagation
```cpp
LogicalResult getConvertBackwardSlice(
    OpOperand &root, SetVector<Value> &slice, Attribute rootEncoding,
    DenseMap<Value, Attribute> &layout,
    std::function<bool(Operation *)> stopPropagation = nullptr);
```

### DPAS Analysis
```cpp
auto dpasAnalysis = ttgi::DPASAnalysisFactory::createDPASAnalysis(mod);
auto result = ttgi::DPASAnalysisFactory::canUseDPAS(funcOp, dpasAnalysis);
// Returns DPASAnalysisResult::True/False/Maybe
```

### Lowering Helpers
```cpp
TritonLLVMOpBuilder b(loc, rewriter);  // Builder with convenience methods
createDeviceFunctionCall(rewriter, fnName, retTy, argTys, args, attrs);
intel::mangle(fnName, argTypes);       // SPIR-V name mangling
```

## 7. Capability Gating Attributes

| Attribute Name (getter) | Module Attr String | Checked By |
|------------------------|-------------------|------------|
| `getSupport2DBlockIOAttrName()` | `ttig.support_2d_block_io` | MaterializeBlockPointer, Pipeline |
| `getSupportDPASAttrName()` | `ttig.support_subgroup_matrix_multiply_accumulate` | AccelerateMatmul (indirectly via DPASAnalysis) |
| `getSupportDPASWithBF8AttrName()` | `ttig.support_subgroup_matrix_multiply_accumulate_bf8` | DPASAnalysisFactory (selects Xe2 vs Xe3P) |
| `getSupportBlockScaleDPASAttrName()` | `ttig.support_subgroup_scaled_matrix_multiply_accumulate` | Block Scale DPAS lowering |
| `getSupportPredicatedIOAttrName()` | `ttig.support_predicated_io` | LoadStoreOpToLLVM |
| `getSupportPrefetch256BAttrName()` | `ttig.support_prefetch_256b` | 2D block prefetch verifier |
| `getIsLTSAttrName()` | `ttig.is_lts` | isSPVBuiltinAvailable() (lowering path selection) |
| `getSupportBFloat16ArithmeticAttrName()` | `ttig.support_bfloat16_arithmetic` | ArithEmulation |
| `getSupportBFloat16ConversionAttrName()` | `ttig.support_bfloat16_conversion` | BF16 conversion lowering |
| `getSupportF8ConversionAttrName()` | `ttig.support_f8_conversion` | FP8 conversion lowering |
| `getSupportF4ConversionAttrName()` | `ttig.support_f4_conversion` | FP4 conversion lowering |
| `getSupport16BitAtomicsAttrName()` | `ttig.support_16bit_atomics` | AtomicOpLowering |

## 8. Complete Pass Inventory

### TTIR Transforms (`lib/Dialect/Triton/Transforms/`)

| Pass | CLI Flag | Pattern | Gate |
|------|----------|---------|------|
| RemoveBoundaryChecks | `triton-intel-remove-boundary-checks` | Manual walk | — |
| RemoveMasks | `triton-intel-remove-masks` | Manual walk + TypeSwitch | — |
| StrideVersioning | `triton-intel-stride-versioning` | Manual walk + if-conversion | — |
| BlockPointerToTensorDesc | `triton-intel-block-pointer-to-tdesc` | Manual pattern matching | — |
| TensorDescToBlockPointer | `triton-intel-tdesc-to-block-pointer` | Manual pattern matching | — |
| RewriteTensorDescToPointer | `triton-intel-rewrite-tensor-descriptor-to-pointer` | Manual walk | — |
| FuseReshape | `triton-intel-fuse-reshape` | Manual pattern matching | — |
| SimplifySignedArithmetic | `triton-intel-simplify-signed-arithmetic` | Manual pattern matching | — |

### TTGIR Transforms (`lib/TritonIntelGPUTransforms/`)

| Pass | CLI Flag | Pattern | Gate |
|------|----------|---------|------|
| Coalesce | `tritonintelgpu-coalesce` | Manual walk + axis analysis | — |
| AccelerateMatmul | `tritonintelgpu-accelerate-matmul` | OpRewritePattern (greedy) | DPASAnalysis |
| MaterializeBlockPointer | `tritonintelgpu-materialize-block-pointer` | Manual walk + annotation | `support_2d_block_io` |
| OptimizeDotOperands | `tritonintelgpu-optimize-dot-operands` | OpRewritePattern (greedy) | — |
| RemoveLayoutConversions | `tritonintelgpu-remove-layout-conversions` | Multi-stage (walk + greedy) | — |
| ReduceDataDuplication | `tritonintelgpu-reduce-data-duplication` | OpRewritePattern (greedy) | — |
| Pipeline | `tritonintelgpu-pipeline` | Loop transform + options | `support_2d_block_io` |
| ReduceVariableLiveness | `tritonintelgpu-reduce-variable-liveness` | Manual walk + reorder | — |
| OptimizeReductionLocality | `tritonintelgpu-optimize-reduction-locality` | Manual walk + reshape | — |
| RewriteStackPtr | `tritonintelgpu-rewrite-stack-ptr` | Manual walk + symbol replace | — |
| AnnotateModule | `triton-annotate-module` | Attribute annotation (struct options) | — |

### Conversion/Lowering Passes

| Pass | CLI Flag | Pattern | Source |
|------|----------|---------|--------|
| ConvertTritonIntelGPUToLLVM | `convert-triton-intel-gpu-to-llvm` | Multi-phase ConvertOp + TypeConverter | `lib/TritonIntelGPUToLLVM/` |
| AllocateSharedMemory | `intel-allocate-shared-memory` | Manual walk + metadata | `lib/TritonIntelGPUToLLVM/` |
| ConvertTritonGENToLLVM | `convert-tritongen-to-llvm` | ConvertOpToLLVMPattern | `lib/TritonGENToLLVM/` |
| ConvertTritonGENToSPIRV | `convert-tritongen-to-spirv` | Dialect translation | `lib/TritonGENToSPIRV/` |
| ConvertGPUToTritonGEN | `convert-gpu-to-tritongen` | OpRewritePattern + func call lowering | `lib/GPUToTritonGEN/` |

### Post-Processing (`lib/Target/LLVMIR/`)

| Transform | File | Description |
|-----------|------|-------------|
| LLVMIRFreezeMaskedDivRem | `LLVMIRFreezeMaskedDivRem.cpp` | Insert freeze for masked div/rem |
| PostProcess | `PostProcess.cpp` | Final LLVM IR cleanup |

## 9. Anti-Patterns

### Incorrect Driver Selection
```cpp
// ❌ Using partial conversion for a non-lowering pass
applyPartialConversion(m, target, std::move(patterns));  // Wrong for transforms

// ✅ Use greedy driver for transformation passes
applyPatternsGreedily(m, std::move(patterns));
```

### Premature IR Mutation
```cpp
// ❌ Modifying IR before confirming match
LogicalResult matchAndRewrite(DotOp op, PatternRewriter &rewriter) const override {
  rewriter.eraseOp(someOp);  // Mutation before guards!
  if (!canApply(op))
    return failure();  // IR already damaged
}

// ✅ All guards first, then mutations
LogicalResult matchAndRewrite(DotOp op, PatternRewriter &rewriter) const override {
  if (!canApply(op))
    return failure();
  rewriter.eraseOp(someOp);  // Safe — match confirmed
  return success();
}
```

### Silent Failure
```cpp
// ❌ Ignoring pattern application failure
applyPatternsGreedily(m, std::move(patterns));  // Result ignored

// ✅ Signal pass failure
if (applyPatternsGreedily(m, std::move(patterns)).failed())
  signalPassFailure();
```

### Missing Dependent Dialects
```cpp
// ❌ Creating ops from undeclared dialects causes assertion failures at runtime
// The .td file MUST list all dialects whose ops might be created:
let dependentDialects = [
  "mlir::triton::TritonDialect",
  "mlir::triton::gpu::intel::TritonIntelGPUDialect",
  "mlir::arith::ArithDialect"
];
```

### Missing Capability Gate
```cpp
// ❌ Generating 2D block ops without checking hardware support
void runOnOperation() override {
  // Directly generating 2D block ops — will fail on unsupported hardware
}

// ✅ Check module attribute first
void runOnOperation() override {
  if (!mod->hasAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
    return;
  // Safe to generate 2D block ops
}
```

### Bypassing PatternRewriter
```cpp
// ❌ Direct IR mutation outside rewriter (breaks driver state tracking)
op->erase();
op->setAttr("foo", attr);

// ✅ Use rewriter API
rewriter.eraseOp(op);
rewriter.modifyOpInPlace(op, [&] { op->setAttr("foo", attr); });
```

### Mutable Pass State
```cpp
// ❌ State persists across invocations (breaks thread safety)
struct MyPass {
  DenseMap<Operation *, Value> cache;  // Mutable member!
  void runOnOperation() override {
    cache[op] = val;  // Accumulates across calls
  }
};

// ✅ All state local to runOnOperation
void runOnOperation() override {
  DenseMap<Operation *, Value> cache;  // Local — fresh each invocation
}
```
