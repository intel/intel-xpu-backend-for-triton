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
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUOpInterfaces.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUAttrDefs.td"
include "intel/include/Dialect/TritonIntelGPU/IR/TritonIntelGPUDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

def L2Cache : Resource<"::mlir::triton::gpu::intel::L2Cache">;
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

//===----------------------------------------------------------------------===//
// TritonIntelGPU op definitions
//===----------------------------------------------------------------------===//

class TTIG_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonIntelGPU_Dialect, mnemonic, !listconcat(traits, [VerifyMemDescLayoutsTrait])>;

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
  MemoryEffects<[MemWrite<L2Cache>]>,
  DeclareOpInterfaceMethods<PredicatedOpInterface>,
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
  let arguments = (ins
    Arg<TT_PtrLike, "Global memory pointer", [MemRead]>:$ptr,
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


def TTIG_DescriptorPrefetchOp : TTIG_Op<"descriptor_prefetch", [MemoryEffects<[MemWrite<L2Cache>]>, TT_DescriptorOpInterface]> {
  let summary = "Tensor descriptor prefetch operation";
  let description = [{
    The `descriptor_prefetch` operation prefetches data using a tensor descriptor.
    Example:
      ```mlir
      ttig.descriptor_prefetch %desc[%x, %y] cacheModifier = none evictionPolicy = normal
          : !tt.tensor_desc<tensor<256x32xf16>>
      ```
  }];
  let arguments = (ins
    Arg<TT_TensorDescType, "Global memory pointer", [MemRead]>:$desc,
    Variadic<I32>:$indices,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<TT_EvictionPolicyAttr, "::mlir::triton::EvictionPolicy::NORMAL">:$evict
  );
  let results = (outs);
  let assemblyFormat = [{
    $desc `[` $indices `]`
    oilist(
      `cacheModifier` `=` $cache |
      `evictionPolicy` `=` $evict
    )
    attr-dict `:` qualified(type($desc))
  }];
  let hasVerifier = 1;
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


def TTIG_ExtractDescOp : TTIG_Op<"extract_desc", [Pure]> {
  let summary = "Extract a field from a tensor descriptor struct";
  let description = [{
    Extracts a field from a tensor descriptor by struct index. The descriptor
    struct layout is { shapes[rank], strides[rank], base_ptr }:
      - index 0..rank-1: shapes (i64)
      - index rank..2*rank-1: strides (i64)
      - index 2*rank: base pointer (tt.ptr)

    This allows the LowerTo2DBlockLoad pass to obtain runtime field values
    from loop-carried descriptors without statically tracing to the defining
    MakeTensorDescOp.
  }];

  let arguments = (ins TT_TensorDescType:$desc, I32Attr:$index);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$desc `[` $index `]` attr-dict `:` type($desc) `->` type($result)";
}

def TTIG_Subgroup2DBlockLoadOp : TTIG_Op<"2d_block_load"> {
  let summary = "2D block load from a descriptor-based surface";
  let description = [{
    Loads a 2D tile from a memory surface described by decomposed parameters.
    Created from `tt.descriptor_load` ops where the surface parameters are
    extracted from the `tt.make_tensor_descriptor` that defined the descriptor.

    One `ttig.2d_block_load` may lower to multiple `triton_gen.2Dblockload`
    ops when the tensor requires splitting across warps/registers.

    Tile computation (`tile_height`, `tile_width`, `v_blocks`, `transpose`,
    etc.) is performed during LLVM lowering from the result tensor's encoding.

    The `offset_x` and `offset_y` are surface-level offsets (from the
    descriptor indices) that are combined with per-sub-tile linear layout
    offsets during LLVM lowering.

    When `pad_nan` is set, the LLVM lowering builds boundary masks from
    the surface dimensions and fills out-of-bounds elements with NaN.
    Otherwise, the hardware zero-fills out-of-bounds elements.
  }];

  let arguments = (ins
    Arg<TT_Ptr, "scalar base pointer to the memory surface",
        [MemRead<GlobalMemory>]>:$base_ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$offset_x,
    I32:$offset_y,
    OptionalAttr<UnitAttr>:$pad_nan,
    TTIG_BlockIOMode:$memory_layout
  );

  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $base_ptr `,` $base_width `,` $base_height `,` $base_pitch
    `[` $offset_x `,` $offset_y `]`
    `{` $memory_layout (`,` `pad_nan` $pad_nan^)? `}`
    attr-dict `:` type($base_ptr) `->` type($result)
  }];
  let hasVerifier = 1;
}

def TTIG_Subgroup2DBlockLoadFromPtrOp : TTIG_Op<"2d_block_load_from_ptr", [
  AttrSizedOperandSegments
]> {
  let summary = "2D block load from a tensor of pointers";
  let description = [{
    Loads a 2D tile using a tensor of pointers. Created from `tt.load` ops
    on pointer tensors that are eligible for 2D block I/O.

    Unlike `ttig.2d_block_load` which takes a scalar base pointer and
    surface-level offsets, this op retains the full pointer tensor. During
    LLVM lowering, the per-sub-tile base address is extracted from
    `ptrElems[registerIdx]` and subtraction-based addressing is used:
    the column offset from the linear layout is subtracted from the pointer
    to reach the row start, then passed as the block load's X coordinate.

    One `ttig.2d_block_load_from_ptr` may lower to multiple
    `triton_gen.2Dblockload` ops when the tensor requires splitting across
    warps/registers.

    Tile computation (`tile_height`, `tile_width`, `v_blocks`, `transpose`,
    etc.) is performed during LLVM lowering from the result tensor's encoding.

    Precondition (not verified): The pointer tensor must describe a 2D
    rectangular slice of a strided surface, i.e.
    `ptr[i,j] = base + i*row_stride + j*elem_size` for a constant `base`
    and known `row_stride`. Creating this op from a ptr tensor that does not
    satisfy this precondition produces undefined behavior at lowering.

    `base_pitch` (row stride in bytes) is a runtime `i32` operand, matching
    `ttig.2d_block_load` and `triton_gen.2Dblockload`. A compile-time pitch is
    just an `arith.constant`; a runtime one (e.g. grouped-GEMM `lda`) is the
    computed SSA value.
  }];

  let arguments = (ins
    Arg<TT_PtrLike, "tensor of pointers to the memory surface",
        [MemRead<GlobalMemory>]>:$ptr,
    I32:$base_pitch,
    Optional<TT_BoolLike>:$mask,
    Optional<TT_Type>:$other,
    I32Attr:$base_width,
    I32Attr:$base_height,
    TTIG_BlockIOMode:$memory_layout
  );

  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $ptr `,` $base_pitch (`,` $mask^ `,` $other)?
    ` ` `{` $memory_layout `}`
    attr-dict `:` functional-type(operands, results)
  }];
  let hasVerifier = 1;
}

def TTIG_DescriptorGatherOp : TTIG_Op<"descriptor_gather", [TT_DescriptorLoadLikeOpInterface]> {
  let summary = "gather multiple rows from a descriptor into a single tensor";
  let description = [{
    The `ttig.descriptor_gather` op has the same semantics as `tt.descriptor_gather`.
    Its verifier and layout inference are Intel-specific.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>]>:$desc,
    RankedTensorOf<[I16, I32]>:$x_offsets,
    I32:$y_offset
  );
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $desc `[` $x_offsets `,` $y_offset `]`
    attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TTIG_DescriptorScatterOp : TTIG_Op<"descriptor_scatter", [TT_DescriptorStoreLikeOpInterface]> {
  let summary = "scatter multiple rows to a descriptor from a single tensor";
  let description = [{
    The `ttig.descriptor_scatter` op has same semantic to the `ttg.descriptor_scatter`.
    It's verifier and infer is Intel specific.
  }];

  let arguments = (ins
    Arg<TT_TensorDescType, "", [MemRead<GlobalMemory>, MemWrite<GlobalMemory>]>:$desc,
    RankedTensorOf<[I16, I32]>:$x_offsets,
    I32:$y_offset,
    TT_Tensor:$src
  );

  let assemblyFormat = [{
    $desc `[` $x_offsets `,` $y_offset `]` `,` $src
    attr-dict `:` type(operands)
  }];

  let hasVerifier = 1;
}

#endif
