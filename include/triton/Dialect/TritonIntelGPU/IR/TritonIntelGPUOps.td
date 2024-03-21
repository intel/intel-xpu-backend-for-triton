#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TTIG_PrefetchOp : TTIG_Op<"prefetch",
                      [MemoryEffects<[MemWrite, MemRead]>
                                      ]> {
    let summary = "prefetch from a tensor pointer";

    let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                         TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);

    let results = (outs);

    let assemblyFormat = [{
      $ptr attr-dict `:` type($ptr)
    }];

    let hasCanonicalizer = 0;
}

#endif
