#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
// include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
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
    For example:
      ```mlir
      %0 = triton_intel_gpu.alloc() : <8x64xf32, 3>
      ```
    allocates a `8x64xf32` memory region in local memory.
  }];
  let results = (outs TT_Ptr:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def TTIG_PrefetchOp : TTIG_Op<"prefetch", []> {
  let summary = "Prefetch from a tensor pointer";
  let description = [{
    The `prefetch` operation prefetches a two dimensional input tensor.
    For example:
      ```mlir
      triton_intel_gpu.prefetch %ptr {cache=none, evict=normal, isVolatile = false} : !tt.ptr<tensor<256x32xf16>
      ```
  }];
  let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                       TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);
  let results = (outs);
  let assemblyFormat = [{
    operands attr-dict `:` type($ptr)
  }];
}

def TT_TensorOrTensorPtr : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_GlueOp : TTIG_Op<"glue", [// operands have same type: SameOperandsType,
                               Pure]> {
  let summary = "glue/concat opearands to a larger size";
  let description = [{
    The `glue` operation concatenates its operands tensors to form a larger tensor.
    For example:
      ```mlir
      %val = triton_intel_gpu.glue %tensor1, %tensor2 .... TODO
      ```
    TODO: any constrainsts? Add verification ? Do operands need to have the same shape ?
  }];
  let arguments = (ins Variadic<TT_TensorOrTensorPtr>:$operands);
  let results = (outs TT_TensorOrTensorPtr:$res);
  let assemblyFormat = [{
    operands attr-dict `:` type($operands) `->` type($res)
  }];
}

def TTIG_ExtractOp : TTIG_Op<"extract", [Pure]> {
  let summary = "Extract a value from a tensor";
  let description = [{
    The `extract` operation extracts a subtensor from a $base input tensor. The value extracted has
    shape as specified by the result type. The $idx attribute indicates which portion of the imput
    tensor to extract.
    For example, given:
     ```mlir
      %val = triton_intel_gpu.extract %tensor {idx = 4 : i32} : tensor<32x32xf16> -> tensor<8x16xf16>
      ```
    the result tensor corrsponds to to the fourth 8x16 tile obtain by partitioning the input tensor
    in row major order.

    TODO: is there a better way to describe the semeantics?
  }];
  let arguments = (ins TT_TensorOrTensorPtr:$base, I32Attr:$idx);
  let results = (outs TT_TensorOrTensorPtr:$res);
  let assemblyFormat = [{
    operands attr-dict `:` type($base) `->` type($res)
  }];
}

def TTIG_CastOp : TTIG_Op<"cast", [Pure]> {
  let summary = "Bitcast a tensor";
  let description = [{
    The `cast` operation performs a bitcast of a $src input tensor to another $dst tensor.
    The destination tensor must have the same overall bits and the input tensor.
    Example:
      ```mlir
      %val = triton_intel_gpu.cast : tensor<32x16xf16> -> tensor<32x16xbf16>
      ```
    TODO: document the semantic of the operation.
      can we cast from a element type to another with the same bit width?
      can we use the operation to transpose?
      can we reduce/increase  the element size and change the shape?
  }];
  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$dst);
  let assemblyFormat = [{
    operands attr-dict `:` type($src) `to` type($dst)
  }];
}

#endif
