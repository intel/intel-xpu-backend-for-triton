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
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TT_TensorOrTensorPtr : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_AllocOp : TTIG_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory allocation operation";
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

def TTIG_CastOp : TTIG_Op<"cast", 
  [Pure,
   DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Tensor bitcast operation";
  let description = [{
    The `cast` operation performs a bitcast of a $src input tensor to another $dst tensor.
    The destination tensor must have the same overall bits and the input tensor.
    Example:
      ```mlir
      %val = triton_intel_gpu.cast : tensor<32x16xf16> -> tensor<32x16xbf16>
      ```
      TODO: should use the tensor.cast operation ? 
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

def TTIG_ConcatOp : TTIG_Op<"concat", [Pure]> {
  let summary = "Tensor concatenation operation";
  let description = [{
    The `concat` operation concatenates its input operands along a specified dimension.
    The concatenated result must have size along the concatenated dimension equal to 
    the sum of the input sizes along that dimension. The other dimensions in all inputs 
    and in the results must be equal.

    Examples:
      ```mlir
      %val = triton_intel_gpu.glue %tensor1, %tensor2 {dim = 0: i32} : 
           (tensor<16x8xf16>, tensor<8x8xf16>) -> tensor<24x8xf16>
      %ptr = triton_intel_gpu.glue %ptr1, %ptr2 {dim = 1: i32} : 
           (!tt.ptr<tensor<16x8xf16>>, !tt.ptr<tensor<16x16xf16>>) -> !tt.ptr<tensor<16x24xf16>>

      ```
    TODO: Add verification.
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

def TTIG_PrefetchOp : TTIG_Op<"prefetch", []> {
  let summary = "Tensor prefetch operation";
  let description = [{
    The `prefetch` operation prefetches a two dimensional input tensor.
    For example:
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

#endif
