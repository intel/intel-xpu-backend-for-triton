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

include "triton/Dialect/TritonGEN/IR/TritonGENDialect.td"
include "triton/Dialect/TritonGEN/IR/TritonGENAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// TritonGEN op definitions
//===----------------------------------------------------------------------===//

class TritonGEN_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<TritonGEN_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Thread and Block index
//===----------------------------------------------------------------------===//

def IndexType : AnyTypeOf<[I32, I64]>;

def TritonGEN_ThreadIdXOp : TritonGEN_Op<"workitem.id.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_ThreadIdYOp : TritonGEN_Op<"workitem.id.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_ThreadIdZOp : TritonGEN_Op<"workitem.id.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

def TritonGEN_BlockIdXOp : TritonGEN_Op<"workgroup.id.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_BlockIdYOp : TritonGEN_Op<"workgroup.id.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_BlockIdZOp : TritonGEN_Op<"workgroup.id.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Block and Grid range
//===----------------------------------------------------------------------===//

def TritonGEN_BlockDimXOp : TritonGEN_Op<"workgroup.dim.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_BlockDimYOp : TritonGEN_Op<"workgroup.dim.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_BlockDimZOp : TritonGEN_Op<"workgroup.dim.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

def TritonGEN_GridDimXOp : TritonGEN_Op<"grid.dim.x", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_GridDimYOp : TritonGEN_Op<"grid.dim.y", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}
def TritonGEN_GridDimZOp : TritonGEN_Op<"grid.dim.z", [Pure]> {
  let arguments = (ins);
  let results = (outs IndexType:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Subgroup index
//===----------------------------------------------------------------------===//

def TritonGEN_SubgroupIdOp : TritonGEN_Op<"subgroup.id", [Pure]> {
  let summary = "Subgroup Index";
  string baseDescription = [{
    The `gen.subgroup.id` operation returns the subgroup ID which is a number
    from 0 to the number of subgroups minus one.
  }];
  let arguments = (ins);
  let results = (outs I32:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TritonGEN_BarrierOp : TritonGEN_Op<"barrier"> {
  let summary = "Workgroup barrier";
  string baseDescription = [{
    The `gen.barrier` operation performs a workgroup barrier and ensures all
    outstanding memory transaction using local or global memory are complete.
  }];
  let arguments = (ins);
  let results = (outs);
  let assemblyFormat = "attr-dict";
}

def TritonGEN_SplitBarrierSignalOp : TritonGEN_Op<"split_barrier_signal"> {
  let summary = "Split barrier signal";
  string baseDescription = [{
    The `gen.split_barrier_signal` operation signals the arrival of a workgroup
    thread at the current program point; the issuing thread continue execution.
    Once all threads in a workgroup have signaled their arrival, any other thread
    waiting at the `gen.split_barrier_wait` operation can exit it.
    The '$mem_fence' attribute is a bitfield that specifies the memory address
    spaces to apply the memory ordering constraints. The '$mem_scope' attribute
    describes the work-items to apply the memory ordering constraints.
    Behavior is undefined:
      - unless all threads in a workgroup participate in the barrier
      - a thread waits on a barrier before signaling
      - the '$mem_fence' and '$mem_scope' attributes aren't the same for all
        threads in a workgroup
    Furthermore, if the `$mem_fence` argument differs between the barrier signal
    and wait operation, then only memory operations for the address spaces specified
    by the intersection of the two flags arguments are visible.
  }];
  let arguments = (ins TritonGEN_MemFence:$mem_fence, TritonGEN_MemScope:$mem_scope);
  let results = (outs);
  let assemblyFormat = [{
    ` ` `{` `mem_fence` `=` $mem_fence `,` `mem_scope` `=` $mem_scope `}` attr-dict
  }];
}

def TritonGEN_SplitBarrierWaitOp : TritonGEN_Op<"split_barrier_wait"> {
  let summary = "Split barrier wait";
  string baseDescription = [{
    The `gen.split_barrier_wait` operation blocks the issuing workgroup thread
    until all other threads arrive at the corresponding `gen.split_barrier_arrive`
    operation. Please refer to the `gen.split_barrier_signal` documentation for the
    description of the attributes and the constrains on the operation.
  }];
  let arguments = (ins TritonGEN_MemFence:$mem_fence, TritonGEN_MemScope:$mem_scope);
  let results = (outs);
  let assemblyFormat = [{
    ` ` `{` `mem_fence` `=` $mem_fence `,` `mem_scope` `=` $mem_scope `}` attr-dict
  }];
}

def TritonGEN_NamedBarrierSignalOp : TritonGEN_Op<"named_barrier_signal">,
  Arguments<(ins I32:$barrier_id, I32:$thread_group_count)> {
  let summary = "Named barrier signal";

  string baseDescription = [{
    The `gen.named_barrier_signal` operation signals the arrival of a thread at
    the current program point. The '$barrier_id' parameter is an integer value
    which identifies the named barrier. The '$thread_group_count' parameter
    specifies the number of threads participating in the barrier. Once all
    participating threads have signaled their arrival, any other participating
    thread waiting at the `gen.named_barrier_wait` operation associated with the
    same '$barrier_id' can exit the wait operation.
  }];

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)`
  }];
}

def TritonGEN_NamedBarrierWaitOp : TritonGEN_Op<"named_barrier_wait">,
  Arguments<(ins I32:$barrier_id)> {
  let summary = "Named barrier wait";

  string baseDescription = [{
    The `gen.named_barrier_wait` operation blocks threads participating in the
    named barrier operation identified by '$barrier_id' until all participating
    threads have reached the `gen.named_barrier_signal` operation associated with
    the same '$barrier_id'.
  }];

  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

def IntegerOrFloatType : AnyTypeOf<[AnyInteger, AnyFloat]>;

def TritonGEN_SubGroupShuffleOp : TritonGEN_Op<"sub_group_shuffle", [
      TypesMatchWith<"result and value have the same type",
                     "res", "value", "$_self">]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 TritonGEN_ShflKindAttr:$kind)> {
  let summary = "Subgroup shuffle";

  string baseDescription = [{
    The `gen.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    subgroup local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated subgroup local ID. It returns the associated
    $value for the work item with subgroup local ID equal to:
    - $kind == xor, the current invocation’s subgroup local ID xor’ed with $mask.
    - $kind == up, the current invocation’s subgroup local ID - $mask.
    - $kind == down, the current invocation’s subgroup local ID + $mask.
    - $kind == idx, the subgroup local ID $mask.
  }];

  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($value) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

class FixedVectorOfRankAndType<list<int> allowedRanks,
                               list<Type> allowedTypes> : AllOfType<
  [FixedVectorOf<allowedTypes>, FixedVectorOfRank<allowedRanks>],
  FixedVectorOf<allowedTypes>.summary #
  FixedVectorOfRank<allowedRanks>.summary,
  "::mlir::VectorType">;

def TritonGEN_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def TritonGEN_MatrixDPASOp : TritonGEN_Op<"dpas">,
  Results<(outs FixedVectorOf<[TritonGEN_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [TritonGEN_MatrixElemType]>:$b,
    TritonGEN_PrecisionTypeAttr:$pa,
    TritonGEN_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "Matrix multiply-add";

  string baseDescription = [{
    The `gen.dpas` operation is a matrix multiplication plus accumulation:

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

def TritonGEN_Matrix2DBlockLoadOp : TritonGEN_Op<"2Dblockload">,
  Results<(outs FixedVectorOf<[TritonGEN_MatrixElemType]>:$res)>,
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
    I1Attr:$vnni_transform
  )> {

  let summary = "2D block load";

  string baseDescription = [{
    The `gen.2Dblockload` operation loads a two dimensional matrix tile from a larger matrix residing
    in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to load
      $base_width, $base_height, $base_pitch - the shape of matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to load
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to load
      $transpose - transpose the tile in registers (useful for 32 bit element type)
      $vnni_transform - transpose and pack the submatrix in registers (useful for < 32 bit element types)

    Notes:
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand of a `gen.tile` operations
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
      - if the tile loaded contains out of bound elements of the matrix, they are filled with 0.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
    `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
    `vnni_transform` `=` $vnni_transform `}` attr-dict `:` functional-type(operands, results)
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
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    FixedVectorOf<[TritonGEN_MatrixElemType]>:$stored_val
  )> {

  let summary = "2D block store";

  string baseDescription = [{
    The `gen.2Dblockstore` operation stores a two dimensional tile into a larger matrix residing
    in memory. The parameters are:
      $ptr - the base address of the matrix where to store the tile
      $base_width, $base_height, $base_pitch - the shape of the matrix
      $x, $y, $tile_width, $tile_height - the starting offsets and shape of the tile to store
      $elem_size_in_bits - the size in bits of the matrix element
        - 32 for f32, bf32
        - 16 for f16, int16, bf16
        - 8 for int8, int4, int2
      $v_blocks - number of tiles to store
      $transpose - transpose the tile in registers (useful for 32 bit element types)
      $vnni_transform - transpose and pack the tile in registers (useful for < 32 bit element types)
      $stored_val - the tile to store

    Notes:
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand of a `gen.tile` operations
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
      `vnni_transform` `=` $vnni_transform `}` attr-dict `:` `(` type(operands) `)`
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
    I1Attr:$transpose,
    I1Attr:$vnni_transform,
    TritonGEN_PrefetchCacheControl:$cache_control
  )> {

  let summary = "2D block prefetch";

  string baseDescription = [{
    The `gen.2Dblockprefetch` operation prefetches a two dimensional tile from a larger matrix residing
    in memory. The parameters are:
      $ptr - the base address of the matrix containing the tile to prefetch
      $base_width, $base_height, $base_pitch - the shape of the matrix
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of tile to prefetch
    $elem_size_in_bits - the size in bits of the matrix element
      - 32 for f32, bf32
      - 16 for f16, int16, bf16
      - 8 for int8, int4, int2
    $v_blocks - number of tiles to prefetch
    $transpose - transpose the tile in registers (useful for 32 bit element types)
    $vnni_transform - transpose and pack the tile in registers (useful for < 32 bit element types)
    $cache_control - an enumerator that sets the L1 and L3 cache behaviour

    Notes:
      - the $transpose and $vnni_transform parameters are mutual exclusive
      - transposing the tile loaded is typically used for the B matrix operand of a `gen.tile` operations
        (D = C + A * B), where A has row-major layout in registers and B should have column-major layout.
  }];

  let assemblyFormat = [{
    operands ` ` `{` `elem_size_in_bits` `=` $elem_size_in_bits `,` `tile_width` `=` $tile_width `,`
      `tile_height` `=` $tile_height `,` `v_blocks` `=` $v_blocks `,` `transpose` `=` $transpose `,`
      `vnni_transform` `=` $vnni_transform `,` `cache_control` `=` $cache_control `}` attr-dict `:`
      `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

#endif // TRITONGEN_OPS
