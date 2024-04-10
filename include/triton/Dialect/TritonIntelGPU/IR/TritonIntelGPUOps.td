#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TT_TensorOrTensorPtr : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

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

def TTIG_ConcatOp : TTIG_Op<"concat", [Pure]> {
  let summary = "Tensor concatenation operation";
  let description = [{
    The `concat` operation concatenates its input operands along a specified
    dimension. The concatenated result must have size along the concatenated
    dimension equal to the sum of the input sizes along that dimension. The
    other dimensions in all inputs and in the results must be equal.

    Examples:
      ```mlir
      %val = triton_intel_gpu.concat %tensor1, %tensor2 {dim = 0: i32} :
           (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
      %ptr = triton_intel_gpu.concat %ptr1, %ptr2 {dim = 1: i32} :
           (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf16>>)
           -> !tt.ptr<tensor<16x24xf16>>

      ```
  }];
  let arguments = (ins Variadic<TT_TensorOrTensorPtr>:$operands, I32Attr:$dim);
  let results = (outs TT_TensorOrTensorPtr:$res);
  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TTIG_ExtractOp : TTIG_Op<"extract", [Pure]> {
  let summary = "Tensor extract operation";
  let description = [{
    The `extract` operation extracts a subtensor (or ptr to subtensor) from an
    input tensor (or ptr to tensor) as specified by the given index.
    The value extracted has shape as specified by the result type and the same
    element type as the input tensor.

    Example:
     ```mlir
      %val = triton_intel_gpu.extract %tensor1[2] : tensor<32x32xf16>
           -> tensor<16x16xf16>
      %ptr = triton_intel_gpu.extract %ptr1[2] : !tt.ptr<tensor<32x32xf16>>
           -> !tt.ptr<tensor<16x16xf16>>
      ```

    In the examples above the result tensor (ptr to tensor) is obtained by:
      - partitioning the input shape into subtensors (ptr to subtensors) having
        shape equal to the result shape (i.e. <16x16xf16>)
      - taking the 3rd subtensor (as indicated by the index) counting in
        row-major order
  }];
  let arguments = (ins TT_TensorOrTensorPtr:$operand, I32Attr:$index);
  let results = (outs TT_TensorOrTensorPtr:$res);
  let assemblyFormat = [{
    $operand `[` $index `]` attr-dict `:` type($operand) `->` type($res)
  }];
  let hasVerifier = 1;
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
