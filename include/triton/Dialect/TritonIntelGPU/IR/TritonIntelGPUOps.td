#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" 

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TTIG_AllocOp : TTIG_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory allocation operation"; 

  let description = [{
    The `alloc` operation allocates a region of memory, as specified by its type.

    Example:

    ```mlir
    %0 = triton_intel_gpu.alloc() : <8x64xf32, 3>
    ```
  }];

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];

  let results = (outs TT_Ptr:$result);
}

def TTIG_PrefetchOp : TTIG_Op<"prefetch", []> {
  let summary = "Prefetch from a tensor pointer";

  let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                       TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);

  let results = (outs);

  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

#endif
