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

//===----------------------------------------------------------------------===//
// TritonGEN op definitions
//===----------------------------------------------------------------------===//

class TritonGEN_Op<string mnemonic, list<Trait> traits = []> :
  Op<TritonGEN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Subgroup index
//===----------------------------------------------------------------------===//

def TritonGEN_SubgroupIdOp : TritonGEN_Op<"subgroup.id", [Pure]> {
  let summary = "Subgroup Index";
  let description = [{
    The `triton_gen.subgroup.id` operation returns the subgroup ID which is a
    number from 0 to the number of subgroups minus one.
  }];
  let arguments = (ins);
  let results = (outs I32:$res);
  let assemblyFormat = [{
    attr-dict `:` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Subgroup local index
//===----------------------------------------------------------------------===//

def TritonGEN_SubgroupLocalIdOp : TritonGEN_Op<"subgroup.local.id", [Pure]> {
  let summary = "Subgroup Local Invocation Index";
  string baseDescription = [{
    The `triton_gen.subgroup.local.id` operation returns the local invocation ID
    of the work-item within the sub-group, which is a number in the range `[0,
    subgroup_size)`.
  }];
  let arguments = (ins);
  let results = (outs I32 : $res);
  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def TritonGEN_SplitBarrierSignalOp : TritonGEN_Op<"split_barrier_signal"> {
  let summary = "Split barrier signal";
  let description = [{
    The `triton_gen.split_barrier_signal` operation signals the arrival of a
    workgroup thread at the current program point; the issuing thread continue
    execution.
    Once all threads in a workgroup have signaled their arrival, any other thread
    waiting at the `triton_gen.split_barrier_wait` operation can exit it.
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
  let description = [{
    The `triton_gen.split_barrier_wait` operation blocks the issuing workgroup
    thread until all other threads arrive at the corresponding
    `triton_gen.split_barrier_arrive` operation. Please refer to the
    `triton_gen.split_barrier_signal` documentation for the description of the
    attributes and the constrains on the operation.
  }];
  let arguments = (ins TritonGEN_MemFence:$mem_fence, TritonGEN_MemScope:$mem_scope);
  let results = (outs);
  let assemblyFormat = [{
    ` ` `{` `mem_fence` `=` $mem_fence `,` `mem_scope` `=` $mem_scope `}` attr-dict
  }];
}

def TritonGEN_SubGroupReduceOp : TritonGEN_Op<"sub_group_reduce", [
      AllTypesMatch<["res", "value"]>]>,
  Results<(outs SignlessIntegerOrFloatLike:$res)>,
  Arguments<(ins SignlessIntegerOrFloatLike:$value,
                 TritonGEN_ReduceKindAttr:$kind,
                 I32Attr:$size)> {
  let summary = "Subgroup reduce";

  let description = [{
    The `triton_gen.sub_group_reduce` operation is invoked by all work items in
    a subgroup, each of them providing a $value. The $size argument is used to
    form groups of $size consecutive work items called clusters. Each cluster
    performs the reduction operation identified by $kind. The result of the
    cluster reduction is propagated to the work items belonging to that cluster.
  }];

  let assemblyFormat = [{
    $kind $value ` ` `{` `size` `=` $size `}` attr-dict `:` type($value)
  }];

  let hasVerifier = 1;
}

def TritonGEN_SubGroupScanOp : TritonGEN_Op<"sub_group_scan", [
      AllTypesMatch<["res", "value"]>]>,
  Results<(outs SignlessIntegerOrFloatLike:$res)>,
  Arguments<(ins SignlessIntegerOrFloatLike:$value,
                 TritonGEN_ReduceKindAttr:$reduce_kind,
                 TritonGEN_ScanKindAttr:$scan_kind)> {
  let summary = "Subgroup scan";

  let description = [{
    The `triton_gen.sub_group_scan` operation is invoked by all work items in
    a subgroup, each of them providing a $value. Each work item performs the
    reduction operation identified by $reduce_kind. The $scan_kind attribute
    indicates whether to perform an inclusive or exclusive scan. The result
    of the scan operation is returned for each work item.
    Note: The scan order is defined by increasing sub-group local ID within
    the sub-group.
  }];

  let assemblyFormat = [{
    $reduce_kind $value ` ` `{` `kind` `=` $scan_kind `}` attr-dict `:` type($value)
  }];
}

//===----------------------------------------------------------------------===//
// Matrix operations
//===----------------------------------------------------------------------===//

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
    FixedVectorOf<[TritonGEN_MatrixElemType]>:$stored_val,
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

def TritonGEN_SIMDBlockReadOp: TritonGEN_Op<"simdblockread">,
  Results<(outs FixedVectorOf<[AnyTypeOf<[AnyI8, AnyI16, AnyI32, AnyI64]>]>:$res)>,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemRead]>:$ptr
  )> {

  let summary = "simd block read";

  let description = [{
    The `triton_gen.simdblockread` operation performs simd block read from
      a start address without laneId offset. The parameters are:
      $ptr - the base address to read data
  }];

  let assemblyFormat = [{
    operands ` ` attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def TritonGEN_SIMDBlockWriteOp : TritonGEN_Op<"simdblockwrite">,
  Arguments<(ins
    Arg<LLVM_AnyPointer, "", [MemWrite]>:$ptr,
    FixedVectorOf<[AnyTypeOf<[AnyI8, AnyI16, AnyI32, AnyI64]>]>:$val
  )> {

  let summary = "simd block write";

  let description = [{
    The `triton_gen.simdblockwrite` operation performs simd block write to
      a start address without laneId offset. The parameters are:
      $ptr - the base address to be written
      $val - the value vector to write
  }];

  let assemblyFormat = [{
    operands ` ` attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}
#endif // TRITONGEN_OPS
