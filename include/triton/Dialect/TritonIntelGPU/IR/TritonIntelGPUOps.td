//===- TritonIntelGPUOps.td - TritonIntelGPU op defs -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_INTEL_GPU_OPSDEFS
#define TRITON_INTEL_GPU_OPSDEFS

include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "triton/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TT_TensorOrTensorPtr : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_AllocOp : TTIG_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory as specified by its type.
    Example:
      ```mlir
      %0 = triton_intel_gpu.alloc() : <8x64xf32, 3>
      ```
  }];
  let results = (outs TT_Ptr:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def TTIG_GlueOp : TTIG_Op<"glue", [Pure]> {
  let summary = "Tensor glueing operation";
  let description = [{
    The `glue` operation glues its input operands to form the result tensor (ptr
    to tensor) shape. Input operands are first concatenated along the first
    (leftmost) dimension until the result shape along that dimension is reached,
    then along the next dimension, and so on. The result tensor and all input
    operands must have the same rank and element type. Furthermore all the input
    tensors must have the same shape. Concatenation of the input operands must
    yield the exact tensor shape of the result.

    Examples:
      ```mlir
      %res1 = triton_intel_gpu.glue %t1, %t2 : (tensor<16x8xf16>, tensor<16x8xf16>)
            -> tensor<32x8xf16>
      %res2 = triton_intel_gpu.glue %t1, %t2, %t3, %t4
            : (tensor<16x8xf16>, tensor<16x8xf16>, tensor<16x8xf16>, tensor<16x8xf16>)
            -> tensor<32x16xf16>
      %res3 = triton_intel_gpu.glue %p1, %p2, %p3
            : (ptr<tensor<16x8xf16>>, ptr<tensor<16x8xf16>>, ptr<tensor<16x8xf16>>)
            -> ptr<tensor<16x24xf16>>
      ```
  }];
  let arguments = (ins Variadic<TT_TensorOrTensorPtr>:$operands);
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
        column-major order
  }];
  let arguments = (ins TT_TensorOrTensorPtr:$base, I32Attr:$index);
  let results = (outs TT_TensorOrTensorPtr:$res);
  let assemblyFormat = [{
    $base `[` $index `]` attr-dict `:` type($base) `->` type($res)
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
}

def TTIG_PrefetchOp : TTIG_Op<"prefetch"> {
  let summary = "Tensor prefetch operation";
  let description = [{
    The `prefetch` operation prefetches an input tensor.
    Example:
      ```mlir
      triton_intel_gpu.prefetch %ptr {cache=none, evict=normal, isVolatile=false}
          : !tt.ptr<tensor<256x32xf16>
      ```
  }];
  let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr, TT_CacheModifierAttr:$cache,
                       TT_EvictionPolicyAttr:$evict, BoolAttr:$isVolatile);
  let results = (outs);
  let assemblyFormat = [{
    operands attr-dict `:` type($ptr)
  }];
}

def TTIG_Load2DOp : TTIG_Op<"load_2d", [
  SameLoadStoreOperandsAndResultShape,
  SameLoadStoreOperandsAndResultEncoding,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches ptr type", "ptr", "result", "getPointeeType($_self)">
]> {
    let summary = "Intel load 2d block";

    let arguments = (ins AnyTypeOf<[TT_TensorPtr]>:$ptr,
                         OptionalAttr<TT_PaddingOptionAttr>:$padding,
                         //DenseI32ArrayAttr:$strides, //only support contiguous 2d load.
                         TT_CacheModifierAttr:$cache,
                         TT_EvictionPolicyAttr:$evict,
                         BoolAttr:$isVolatile);

    let results = (outs TT_Type:$result);

    let builders = [
        // A tensor of pointers or a pointer to a scalar
        OpBuilder<(ins "Value":$ptr, "std::optional<triton::PaddingOption>":$padding, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,

        OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                       "triton::EvictionPolicy":$evict, "bool":$isVolatile)>,
    ];

    let assemblyFormat = "$ptr attr-dict `:` type($ptr) `->` type($result)";

    let hasCanonicalizer = 1;
}

#endif
