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
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, traits>;

def TT_TensorOrTensorPtr : AnyTypeOf<[TT_Tensor, TT_TensorPtr]>;

def TTIG_AllocOp : TTIG_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Memory allocation operation";
  let description = [{
    The `alloc` operation allocates a region of memory as specified by its type.
    Example:
      ```mlir
      %0 = ttig.alloc() : <8x64xf32, 3>
      ```
  }];
  let results = (outs TT_Ptr:$result);
  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def TTIG_PrefetchOp : TTIG_Op<"prefetch", [
  TypesMatchWith<"mask type matches ptr type", "ptr", "mask", "getI1SameShape(getPointeeType($_self))",
                 "($_op.getOperands().size() <= 1) || std::equal_to<>()">,
]> {
  let summary = "Tensor prefetch operation";
  let description = [{
    The `prefetch` operation prefetches an input tensor.
    Example:
      ```mlir
      ttig.prefetch %ptr {cache=none, evict=normal, isVolatile=false}
          : !tt.ptr<tensor<256x32xf16>
      ```
  }];
  let arguments = (
    ins AnyTypeOf<[TT_PtrLike, TT_TensorPtr]>:$ptr,
    Optional<TT_BoolLike>:$mask,
    TT_CacheModifierAttr:$cache,
    TT_EvictionPolicyAttr:$evict,
    BoolAttr:$isVolatile
  );
  let results = (outs);
  let builders = [
    OpBuilder<(ins "Value":$ptr, "triton::CacheModifier":$cache,
                   "triton::EvictionPolicy":$evict, "bool":$isVolatile)>
  ];
  let assemblyFormat = [{
    $ptr (`,` $mask^)? attr-dict `:` type($ptr)
  }];
}

// same as tt.broadcast except that we don't require SameOperandsAndResultEncoding
def TTIG_BroadcastOp : TTIG_Op<"broadcast", [Pure, SameOperandsAndResultElementType]> {
  let summary = "broadcast a tensor";
  let description = [{
    For a given tensor, broadcast changes one or more dimensions with size 1
    to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
    change the size of a non-1 dimension.
  }];

  let arguments = (ins TT_Tensor:$src);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
}

class IsLocalPointerToElementType<string pointer, string value, string summary> :
    TypesMatchWith<summary, value, pointer, [{
      ::mlir::triton::PointerType::get( }] # ElementType<"_self">.result # [{,
          static_cast<unsigned>(::mlir::triton::TritonGEN::kWorkgroup))
    }]>;

// NOTE: This operation shouldn't be needed, as simply modifying the encoding
// would be equivalent. As we are not handling encodings much, I created this for
// prototyping.
def TTIG_SubGroupTransposeOp
    : TTIG_Op<"sub_group_transpose",
        [AllTypesMatch<["src", "res"]>,
         IsLocalPointerToElementType<"local_buffer", "src",
             "local_buffer can be used to store src in local memory">]> {
  let summary = "Sub-group elements transpose operation";
     let description = [{
     For a distribution of tensor elements across a sub-group forming a squared
     matrix like:

    ```
| Sub-group local id |                       Elements                        |
|--------------------|-------------------------------------------------------|
| 0                  | [0,sub_group_size)                                    |
| 1                  | [sub_group_size,2*sub_group_size)                     |
| ...                | ...                                                   |
| sub_group_size-1   | [(sub_group_size-1)*sub_group_size,sub_group_size**2) |
    ```

    Redistribute elements such as the following mapping applies:

    ```
| Sub-group local id |                           Elements                           |
|--------------------|--------------------------------------------------------------|
| 0                  | 0, sub_group_size, 2*sub_group_size, ...                     |
| 1                  | 1, sub_group_size+1, 2*sub_group_size+1, ...                 |
| ...                | ...                                                          |
| sub_group_size-1   | [sub_group_size-1+i*sub_group_size|i in [0, sub_group_size)] |
    ```

    In order to do this transpose, each sub-group stores its contents to local
    memory via `$local_buffer`, each of them forming a matrix there, and loads
    them back transposed. Note no barriers will be needed as no memory is shared
    across sub-groups.

    Also note the tensor shape does not change, as only the encoding of the tensor
    is modified.

    Example:
      ```
%matrix_t = ttig.sub_group_transpose %local_buffer, %src : tensor<256x32xf16>
      ```

    This operation only supports tranposing tensors of shape
    `sub_group_size x sub_group_size`.
   }];
  let arguments = (ins TT_Ptr: $local_buffer, TT_Tensor:$src);
  let results = (outs TT_Tensor:$res);
  let assemblyFormat = "operands attr-dict `:` type($src)";
  let hasVerifier = 1;
}

#endif
