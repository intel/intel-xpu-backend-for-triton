//===-- TritonGENOps.td - TritonGEN op definitions ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the TritonGEN IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONGEN_OPS
#define TRITONGEN_OPS

include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.td"
include "intel/include/Dialect/TritonGEN/IR/TritonGENAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType

//===----------------------------------------------------------------------===//
// TritonGEN op definitions
//===----------------------------------------------------------------------===//

class TritonGEN_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonGEN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TritonGEN_BarrierOp : TritonGEN_Op<"barrier"> {
  let summary = "Workgroup barrier";
  let description = [{
    The `triton_gen.barrier` operation performs a workgroup barrier and ensures
    all outstanding memory transaction using local or global memory are complete.
  }];
  let arguments = (ins TritonGEN_MemFence:$mem_fence);
  let results = (outs);
  let assemblyFormat = "attr-dict";
  let assemblyFormat = [{
    ` ` `{` `mem_fence` `=` $mem_fence `}` attr-dict
  }];
}

def TritonGEN_SplitBarrierArriveOp : TritonGEN_Op<"split_barrier_arrive"> {
  let summary = "Split barrier signal";
  let description = [{
    Indicates that an invocation has arrived at a split control barrier. This
    may allow other invocations waiting on the split control barrier to continue
    executing.

    When `Execution` is `Workgroup` or larger, behavior is undefined unless all
    invocations within `Execution` execute the same dynamic instance of this
    instruction. When `Execution` is `Subgroup` or `Invocation`, the behavior of
    this instruction in non-uniform control flow is defined by the client API.

    If `Semantics` is not `None`, this instruction also serves as the start of a
    memory barrier similar to an `OpMemoryBarrier` instruction with the same
    `Memory` and `Semantics` operands. This allows atomically specifying both a
    control barrier and a memory barrier (that is, without needing two
    instructions). If `Semantics` is `None`, `Memory` is ignored.
  }];
  let arguments = (ins TritonGEN_MemScope:$execution_scope, TritonGEN_MemScope:$memory_scope);
  let results = (outs);
  let assemblyFormat = [{
    ` ` `{` `execution_scope` `=` $execution_scope `,` `memory_scope` `=` $memory_scope `}` attr-dict
  }];
}

def TritonGEN_SplitBarrierWaitOp : TritonGEN_Op<"split_barrier_wait"> {
  let summary = "Split barrier wait";
  let description = [{
    Waits for other invocations of this module to arrive at a split control
    barrier.

    When `Execution` is `Workgroup` or larger, behavior is undefined unless all
    invocations within `Execution` execute the same dynamic instance of this
    instruction. When `Execution` is `Subgroup` or `Invocation`, the behavior of
    this instruction in non-uniform control flow is defined by the client API.

    If `Semantics` is not `None`, this instruction also serves as the end of a
    memory barrier similar to an `OpMemoryBarrier` instruction with the same
    `Memory` and `Semantics` operands. This ensures that memory accesses issued
    before arriving at the split barrier are observed before memory accesses
    issued after this instruction. This control is ensured only for memory
    accesses issued by this invocation and observed by another invocation
    executing within `Memory` scope. This allows atomically specifying both a
    control barrier and a memory barrier (that is, without needing two
    instructions). If `Semantics` is `None`, `Memory` is ignored.
  }];
  let arguments = (ins TritonGEN_MemScope:$execution_scope, TritonGEN_MemScope:$memory_scope);
  let results = (outs);
  let assemblyFormat = [{
    ` ` `{` `execution_scope` `=` $execution_scope `,` `memory_scope` `=` $memory_scope `}` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

def TritonGEN_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def TritonGEN_MatrixDPASOp : TritonGEN_Op<"dpas">,
  Results<(outs FixedVectorOfNonZeroRankOf<[TritonGEN_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$b,
    TritonGEN_PrecisionTypeAttr:$pa,
    TritonGEN_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "Matrix multiply-add";

  let description = [{
    The `triton_gen.dpas` operation is a matrix multiplication plus accumulation:

      D = C + A x B

      where the A, B, C input matrices and the result D have shapes:
        D : MxN
        C : MxN
        A : MxK
        B : KxN

        M : repeat count, must be 1, 2, 4, or 8
        N : fixed execution size, must be 16
        K : depth * OPS_PER_CHAN
            OPS_PER_CHAN
              1 : for TF32
              2 : for 16-bit precision(BF, HF)
              4 : for 8-bit precision (FP8, UB, B)
              8 : for less-then 8 bit precision (U4/S4, U2/S2).

            If depth is 8, K would be 8, 16, 32, or 64 (based on OPS_PER_CHAN).

    $a, $b, $c, $d - matrix A, B, C, D, respectively
    $pa, $pb - precision of matrix A and B resepectively
    $rc - repeat count
  }];

  let assemblyFormat = [{
    operands ` ` `{` `pa` `=` $pa `,` `pb` `=` $pb `,` `rc` `=` $rc `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TritonGEN_MatrixBlockScaleDPASOp : TritonGEN_Op<"bdpas">,
  Results<(outs FixedVectorOfNonZeroRankOf<[F32, BF16]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [F32, BF16]>:$c,
    FixedVectorOfRankAndType<[1], [AnyI16]>:$a,
    FixedVectorOfRankAndType<[1], [AnyI32]>:$b,
    AnyTypeOf<[AnyI8, FixedVectorOfRankAndType<[1], [AnyI8]>]>:$scaleA,
    AnyTypeOf<[AnyI8, FixedVectorOfRankAndType<[1], [AnyI8]>]>:$scaleB,
    TritonGEN_PrecisionTypeAttr:$pa,
    TritonGEN_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "Block scale matrix multiply-add for MXFP";

  let description = [{
    The `triton_gen.bdpas` operation is a matrix multiplication plus accumulation for MXFP:

      D = C + (A x B) * scaleA * scaleB

      where the A, B, C input matrices and the result D have shapes:
        D : MxN
        C : MxN
        A : MxK
        B : KxN
        scaleA: Mx(K/32)
        scaleB: (K/32)xN

        M : repeat count, must be 8
        N : fixed execution size, must be 16
        K : depth * OPS_PER_CHAN
            OPS_PER_CHAN
              2 : for 16-bit precision(BF, HF)
              4 : for 8-bit precision (FP8, UB, B)
              8 : for less-then 8 bit precision (U4/S4, U2/S2).

            If depth is 8, K would be 8, 16, 32, or 64 (based on OPS_PER_CHAN).

    $a, $b, $c, $d - matrix A, B, C, D, respectively
    $scaleA, $scaleB - scale factors for matrix A and B, respectively
    $pa, $pb - precision of matrix A and B respectively
    $rc - repeat count
  }];

  let assemblyFormat = [{
    operands ` ` `{` `pa` `=` $pa `,` `pb` `=` $pb `,` `rc` `=` $rc `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TritonGEN_Matrix2DBlockLoadOp : TritonGEN_Op<"2Dblockload">,
  Results<(outs FixedVectorOfNonZeroRankOf<[TritonGEN_MatrixElemType, AnyI64]>:$res)>,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    DefaultValuedAttr<TritonGEN_LoadCacheControl, "::mlir::triton::TritonGEN::LoadCacheControl::DEFAULT">:$cache_control
  )> {

  let summary = "2D block load";

  let description = [{
    The `triton_gen.2Dblockload` operation loads a two dimensional matrix tile
    from a larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to load
      $base_width, $base_height, $base_pitch - the shape of matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to load
      $elem_size_in_bits - the size in bits of the matrix element
        - 64 for f64, i64
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to load
      $transpose - transpose the tile in registers (useful for 32 bit element type)
      $vnni_transform - transpose and pack the submatrix in registers (useful for < 32 bit element types)
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand of a `gen.tile` operations
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
      - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.
      - coordinate is provided in elements, while width and pitch are provided in bytes.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
    `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
    `vnni_transform` `=` $vnni_transform `,` `cache_control` `=` $cache_control `}` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TritonGEN_Matrix2DBlockStoreOp : TritonGEN_Op<"2Dblockstore">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    FixedVectorOfNonZeroRankOf<[TritonGEN_MatrixElemType, AnyI64]>:$stored_val,
    DefaultValuedAttr<TritonGEN_StoreCacheControl, "::mlir::triton::TritonGEN::StoreCacheControl::DEFAULT">:$cache_control
  )> {

  let summary = "2D block store";

  let description = [{
    The `triton_gen.2Dblockstore` operation stores a two dimensional tile into a
    larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix where to store the tile
      $base_width, $base_height, $base_pitch - the shape of the matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to store
      $elem_size_in_bits - the size in bits of the matrix element
        - 64 for f64, i64
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to store
      $cache_control - an enumerator that sets the L1 and L3 cache behaviour
      $stored_val - the tile to store

    Notes:
      - coordinate is provided in elements, while width and pitch are provided in bytes.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `cache_control` `=` $cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def TritonGEN_Matrix2DBlockPrefetchOp : TritonGEN_Op<"2Dblockprefetch">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
    I32:$base_width,
    I32:$base_height,
    I32:$base_pitch,
    I32:$x,
    I32:$y,
    I32Attr:$elem_size_in_bits,
    I32Attr:$tile_width,
    I32Attr:$tile_height,
    I32Attr:$v_blocks,
    DefaultValuedAttr<TritonGEN_LoadCacheControl, "::mlir::triton::TritonGEN::LoadCacheControl::DEFAULT">:$cache_control
  )> {

  let summary = "2D block prefetch";

  let description = [{
    The `triton_gen.2Dblockprefetch` operation prefetches a two dimensional tile
    from a larger matrix residing in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to prefetch
      $base_width, $base_height, $base_pitch - the shape of the matrix
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of tile to prefetch
    $elem_size_in_bits - the size in bits of the matrix element
      - 64 for f64, i64
      - 32 for f32, bf32
      - 16 for f16, int16, bf16
      - 8 for int8, int4, int2
    $v_blocks - number of tiles to prefetch
    $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - coordinate is provided in elements, while width and pitch are provided in bytes.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `cache_control` `=` $cache_control `}`
      attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

def TritonGEN_SubGroupBlockMemoryAccessElementType
    : AnyTypeOf<[I8, I16, I32, I64],
                "Valid sub-group block memory access element type">;

def TritonGEN_SubGroupBlockMemoryAccessType
    : AnyTypeOf<[TritonGEN_SubGroupBlockMemoryAccessElementType,
                 FixedVectorOfLengthAndType<[2, 4, 8],
                     [TritonGEN_SubGroupBlockMemoryAccessElementType]>,
                 // Vectors of length 16 only allowed for i8 for now.
                 FixedVectorOfLengthAndType<[16], [I8]>],
          "Valid sub-group block memory access type">;

def TritonGEN_SubGroupBlockMemoryAccessPointerType
    : Type<And<[LLVM_AnyPointer.predicate,
                Or<[CPred<"::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self)" #
                          ".getAddressSpace() == " #
                          "static_cast<unsigned>(kCrossWorkgroup)">,
                    CPred<"::llvm::cast<::mlir::LLVM::LLVMPointerType>($_self)" #
                          ".getAddressSpace() == " #
                          "static_cast<unsigned>(kWorkgroup)">]>]>,
          "LLVM pointer in local or global OpenCL address space",
          "::mlir::LLVM::LLVMPointerType">;

def TritonGEN_SubGroupBlockReadOp: TritonGEN_Op<"sub_group_block_read"> {
  let summary = "Sub-group block read.";

  let description = [{
    The `triton_gen.sub_group_block_read` reads a scalar or vector for each
    work-item in the sub-group from pointer `ptr` as a block operation.
    The data is read strided, so the first value is read from:
    ```
    ptr[sub_group_local_id]
    ```
    and the second one is:
    ```
    ptr[sub_group_local_id + sub_group_size]
    ```
    etc.

    `ptr` must be aligned to the size of the element type of `res`.

     Example:
     ```mlir
     %0 = triton_gen.sub_group_block_read %ptr : !llvm.ptr<1> -> vector<4xi32>
     ```
  }];

  let arguments = (ins
      Arg<TritonGEN_SubGroupBlockMemoryAccessPointerType, "", [MemRead]>:$ptr);

  let results = (outs TritonGEN_SubGroupBlockMemoryAccessType:$res);

  let assemblyFormat = [{
    $ptr attr-dict `:` qualified(type($ptr)) `->` type($res)
  }];
}

def TritonGEN_SubGroupBlockWriteOp : TritonGEN_Op<"sub_group_block_write"> {
  let summary = "simd block write";

  let description = [{
    The `triton_gen.sub_group_block_write` writes a scalar or vector for each
    work-item in the sub-group from pointer `ptr` as a block operation.
    The data is read strided, so the first value is written to:
    ```
    ptr[sub_group_local_id]
    ```
    and the second one is:
    ```
    ptr[sub_group_local_id + sub_group_size]
    ```
    etc.

    `ptr` must be aligned to the size of the element type of `res`.

     Example:
     ```mlir
     %0 = triton_gen.sub_group_block_write %ptr, %val : !llvm.ptr<1>, vector<4xi32>
     ```
  }];

  let arguments = (ins
      Arg<TritonGEN_SubGroupBlockMemoryAccessPointerType, "", [MemRead]>:$ptr,
      TritonGEN_SubGroupBlockMemoryAccessType:$val);

  let results = (outs);

  let assemblyFormat = [{
    $ptr `,` $val attr-dict `:` qualified(type($ptr)) `,` type($val)
  }];
}

def TritonGEN_PredicatedLoadOp
    : TritonGEN_Op<"predicated_load"> {
  let summary = "Predicated load operation";
  let description = [{
    The `triton_gen.predicated_load` operation loads a value from memory
    conditionally based on the predicate. If the predicate is true, the value
    is loaded from the specified pointer; otherwise, a default value is used.
  }];
  let arguments = (ins
      Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr,
      I64: $alignment,
      I1:$predicate,
      AnyType:$default_value);
  let results = (outs
      AnyType:$res);
  let assemblyFormat = [{
    $ptr `,` $alignment `,` $predicate `,` $default_value attr-dict `:` qualified(type($ptr)) `,`
    type($alignment) `,` type($predicate) `,` type($default_value) `->` type($res)
  }];
}

def TritonGEN_PredicatedStoreOp
: TritonGEN_Op<"predicated_store"> {
  let summary = "Predicated store operation";
  let description = [{
    The `triton_gen.predicated_store` operation stores a value to memory
    conditionally based on the predicate. If the predicate is true, the value
    is stored to the specified pointer; otherwise, no operation is performed.
  }];
  let arguments = (ins
      Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
      AnyType:$value,
      I64: $alignment,
      I1:$predicate);
  let results = (outs);
  let assemblyFormat = [{
    $ptr `,` $value `,` $alignment `,` $predicate attr-dict `:` qualified(type($ptr)) `,`
    type($value) `,` type($alignment) `,` type($predicate)
  }];
}

def TritonGEN_FToTf32Op
    : TritonGEN_Op<"f_to_tf32", [SameOperandsAndResultType]> {
  let summary = "Rounding instruction from float to tensor float (TF32) data format";

  let description = [{
    The op converts value numerically from
    a 32-bit floating point type to TF32 with rounding to the nearest even.
  }];

  let arguments = (ins LLVM_ScalarOrVectorOf<F32>:$val);
  let results = (outs LLVM_ScalarOrVectorOf<F32>:$res);
  let assemblyFormat = [{
    $val attr-dict `:` type($val)
  }];
}
#endif // TRITONGEN_OPS
