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
  let summary = "Memory allocation";
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
  let description = [{
    The `prefetch` operation prefetches a two dimensional tile of a input tensor.
  }];
  let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                       TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);
  let results = (outs);
  let assemblyFormat = [{
    $ptr attr-dict `:` type($ptr)
  }];
}

def TT_TensorRelated : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_GlueOp : TTIG_Op<"glue", [// operands have same type: SameOperandsType,
                               Pure]> {
  let summary = "glue/concat opearands to a larger size";
  let description = [{
    The `glue` operation concatenates its operands tensors to form a larger tensor.
    TODO: any constrainsts? Add verification ? Do operands need to have the same shape ?
  }];
  let arguments = (ins Variadic<TT_TensorRelated>:$operands);
  let results = (outs TT_TensorRelated:$result);
  let assemblyFormat = [{
    "$operands attr-dict `:` type($operands) `->` type($result)"
  }];
}

def TTIG_ExtractOp : TTIG_Op<"extract", [Pure]> {
  let summary = "Extract a value from a tensor";
  let description = [{
    The `extract` operation extracts the value at index $idx from an $base input tensor.
  }];
  let arguments = (ins TT_TensorRelated:$base, I32Attr:$idx);
  let results = (outs TT_TensorRelated:$result);
  let assemblyFormat = [{
    "$operands attr-dict `:` type($base) `->` type($result)"
  }];
}

def TTIG_CastOp : TTIG_Op<"cast", [Pure]> {
  let summary = "Bitcast a tensor";
  let description = [{
    The `cast` operation performs a bitcast of a $src input tensor to another $dst tensor.
    The destination tensor must have the same overall bits and the input tensor.
  }];
  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$dst);
  let assemblyFormat = [{
    "$operands attr-dict `:` type($src) `to` type($dst)"
  }];
}

#endif
