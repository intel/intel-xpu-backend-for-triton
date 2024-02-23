//===-- GENOps.td - GEN IR dialect op definition file ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GEN IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_OPS
#define GEN_OPS

include "triton/Dialect/GEN/IR/GENDialect.td"
include "triton/Dialect/GEN/IR/GENAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// GEN op definitions
//===----------------------------------------------------------------------===//

class GEN_Op<string mnemonic, list<Trait> traits = []> :
  LLVM_OpBase<GEN_Dialect, mnemonic, traits> {
}

class GEN_DeviceFunctionOp<string mnemonic, string fnName,
                            int arg, list<Trait> traits = []>
  : GEN_Op<mnemonic, !listconcat(traits, [Pure])>,
    Results<(outs LLVM_Type:$res)>, Arguments<(ins)> {
  string llvmBuilder = [{
    llvm::Type *retType = builder.getInt64Ty();
    llvm::Type *argType = builder.getInt32Ty();
    llvm::Value *val = llvm::ConstantInt::get(argType, }] # arg # [{);
    llvm::CallInst *ci = createDeviceFunctionCall(builder, "}] # fnName # [{",
        retType, { argType }, { val });
    llvm::Type *opRetType = cast<llvm::IntegerType>(
        moduleTranslation.convertType(op->getResultTypes()[0]));
    if (opRetType == retType)
      $res = ci;
    else {
      llvm::Instruction::CastOps opcode =
          llvm::CastInst::getCastOpcode(ci, false, opRetType, false);
      $res = builder.CreateCast(opcode, ci, opRetType);
    }
  }];

  let assemblyFormat = "attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Thread index and Block index
//===----------------------------------------------------------------------===//

def GEN_ThreadIdXOp : GEN_DeviceFunctionOp<"workitem.id.x",
                        "_Z12get_local_idj", 0>;
def GEN_ThreadIdYOp : GEN_DeviceFunctionOp<"workitem.id.y",
                        "_Z12get_local_idj", 1>;
def GEN_ThreadIdZOp : GEN_DeviceFunctionOp<"workitem.id.z",
                        "_Z12get_local_idj", 2>;
def GEN_BlockIdXOp : GEN_DeviceFunctionOp<"workgroup.id.x",
                        "_Z12get_group_idj", 0>;
def GEN_BlockIdYOp : GEN_DeviceFunctionOp<"workgroup.id.y",
                        "_Z12get_group_idj", 1>;
def GEN_BlockIdZOp : GEN_DeviceFunctionOp<"workgroup.id.z",
                        "_Z12get_group_idj", 2>;

//===----------------------------------------------------------------------===//
// Block and Grid range
//===----------------------------------------------------------------------===//

def GEN_BlockDimXOp : GEN_DeviceFunctionOp<"workgroup.dim.x",
                        "_Z14get_local_sizej", 0>;
def GEN_BlockDimYOp : GEN_DeviceFunctionOp<"workgroup.dim.y",
                        "_Z14get_local_sizej", 1>;
def GEN_BlockDimZOp : GEN_DeviceFunctionOp<"workgroup.dim.z",
                        "_Z14get_local_sizej", 2>;
def GEN_GridDimXOp : GEN_DeviceFunctionOp<"grid.dim.x",
                        "_Z14get_num_groupsj", 0>;
def GEN_GridDimYOp : GEN_DeviceFunctionOp<"grid.dim.y",
                        "_Z14get_num_groupsj", 1>;
def GEN_GridDimZOp : GEN_DeviceFunctionOp<"grid.dim.z",
                        "_Z14get_num_groupsj", 2>;

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GEN_BarrierOp : GEN_Op<"barrier"> {
  let summary = "Workgroup barrier";

  string baseDescription = [{
    The `gen.barrier` operation performs a workgroup barrier and ensures all
    outstanding memory transaction using local or global memory are complete.
  }];

  string llvmBuilder = [{
    llvm::Type *retType = builder.getVoidTy();
    llvm::Type *argType = builder.getInt32Ty();
    int memFence = 3; // local + global memory fence
    llvm::Value *arg = llvm::ConstantInt::get(argType, memFence);
    createDeviceFunctionCall(builder, "_Z7barrierj", retType, {argType}, {arg},
                             true /*convergent*/);
  }];

  let assemblyFormat = "attr-dict";
}

def IntegerOrFloatType : AnyTypeOf<[AnyInteger, AnyFloat]>;

def GEN_SubGroupShuffleOp : GEN_Op<"sub_group_shuffle", [
      TypesMatchWith<"result and value have the same type",
                     "res", "value", "$_self">]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 GEN_ShflKindAttr:$kind)> {
  let summary = "Subgroup shuffle";
  string baseDescription = [{
    The `gen.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    subgroup local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated subgroup local ID. It returns the associated
    $value for the work item with subgroup local ID equal to:
    - $kind == XOR, the current invocation’s subgroup local ID xor’ed with $mask.
    - $kind == UP, the current invocation’s subgroup local ID - $mask.
    - $kind == DOWN, the current invocation’s subgroup local ID + $mask.
    - $kind == IDX, the subgroup local ID $mask.
  }];
  string llvmBuilder = [{
    $res = createSubGroupShuffle(builder, $value, $mask, $kind);
  }];
  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($value) `->` type($res)
  }];
}

//===----------------------------------------------------------------------===//
// Type Conversions
//===----------------------------------------------------------------------===//

def GEN_FpToFpOp : GEN_Op<"conv.fptofp">,
  Results<(outs AnyFloat:$res)>,
  Arguments<(ins
    AnyFloat:$arg,
    OptionalAttr<GEN_RoundingModeAttr>:$roundingMode
  )> {

  let summary = "Convert between floating point types";

  string baseDescription = [{
    Convert `$arg` to the type of the result `$res`. If necessary, round the
    result according to `$roundingMode`.
  }];

  string llvmBuilder = [{
    $res = createGenISAFpToFp(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    $arg attr-dict `:` type($arg) `to` type($res)
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

def GEN_MatrixElemType : AnyTypeOf<[AnyI8, AnyI16, AnyI32, F32, F16, BF16]>;

def GEN_MatrixDPASOp : GEN_Op<"matrix.dpas">,
  Results<(outs FixedVectorOf<[GEN_MatrixElemType]>:$d)>,
  Arguments<(ins
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$c,
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$a,
    FixedVectorOfRankAndType<[1], [GEN_MatrixElemType]>:$b,
    GEN_PrecisionTypeAttr:$pa,
    GEN_PrecisionTypeAttr:$pb,
    I32Attr:$rc
  )> {

  let summary = "GEN matrix multiply-add (for PVC)";

  string baseDescription = [{
    The 'gen.matrix.dpas' operation is a matrix multiply-add operation as follows:

       D = C + A x B

       where
          D : MxN
          C : MxN
          A : MxK
          B : KxN

          M : repeat count ($rc), must be 1, 2, 4, or 8
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
    $rc - repect count
  }];

  string llvmBuilder = [{
    $d = createGenISADPAS(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GEN_Matrix2DBlockLoadOp : GEN_Op<"matrix.2Dblockload">,
  Results<(outs FixedVectorOf<[GEN_MatrixElemType]>:$res)>,
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

  let summary = "GEN 2D block load";

  string baseDescription = [{
    The 'gen.matrix.2Dblockload' operation loads a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to load
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    $res = createGenISA2DBlockRead(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` functional-type(operands, results)
  }];

  let hasVerifier = 1;
}

def GEN_Matrix2DBlockStoreOp : GEN_Op<"matrix.2Dblockstore">,
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
    FixedVectorOf<[GEN_MatrixElemType]>:$stored_val
  )> {

  let summary = "GEN 2D block store";

  string baseDescription = [{
    The 'gen.matrix.2Dblockstore' operation stores to a submatrix from an array in memory.
    $ptr - the base address of the memory array
    $base_width, $base_height, $base_pitch - the shape of the memory array
    $x, $y, $tile_width, $tile_height - the starting offsets and shape of the submatrix to load
    $elem_size_in_bits - 32 for f32, bf32; 16 for f16, int16, bf16; 8 for int8, int4, int2 and etc
    $v_blocks - number of blocks to store
    $transpose - transpose the submatrix in vector register (useful for 32 bit element types)
    $vnni_transform - transpose and pack the submatrix in register (useful for < 32 bit element types)
    $stored_val - block to store

    The $transpose and $vnni_transform are mutual exclusive. These transformations are used for
    the B matrix in MMA (DPAS) operations D = C + A * B, where A should have row-major layout in register
    and B should have column-major layout.

    If the submatrix contains out of bound elements of the memory array, they are filled with 0.
  }];

  string llvmBuilder = [{
    createGenISA2DBlockWrite(op, builder, moduleTranslation);
  }];

  let assemblyFormat = [{
    operands attr-dict `:` `(` type(operands) `)`
  }];

  let hasVerifier = 1;
}

#endif // GEN_OPS
