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
include "mlir/Interfaces/SideEffectInterfaces.td"

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

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

def TTIG_PrefetchOp : TTIG_Op<"prefetch", []> {
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

#endif
