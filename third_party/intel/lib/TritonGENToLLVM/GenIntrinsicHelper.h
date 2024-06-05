//===- GenIntrinsicHelper.h - Gen intrinsic helper ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

#include "GenIntrinsics.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

mlir::LLVM::LLVMFuncOp
appendOrGetGenISADeclaration(OpBuilder &builder, llvm::GenISAIntrinsic::ID id,
                             ArrayRef<mlir::Type *> mlirTys);

class Intrinsic {
protected:
  LLVM::LLVMFuncOp intrinsicDecl;

public:
  Value operator()(OpBuilder &rewriter, Location loc, ValueRange args) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<LLVM::CallOp>(loc, retType, funName, args);
    funCall.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return funCall.getResult();
  }
};

template <llvm::GenISAIntrinsic::ID INST_ID> class GenISA : public Intrinsic {
public:
  template <typename... OverrideTypes>
  explicit GenISA(OpBuilder &builder, OverrideTypes... retTy) {
    // get GenISA intrinsic declaration.
    intrinsicDecl = appendOrGetGenISADeclaration(builder, INST_ID, {&retTy...});
  }

  template <typename... Args>
  LLVM::CallOp operator()(OpBuilder &rewriter, Location loc, Args... args) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<LLVM::CallOp>(loc, retType, funName,
                                                 ValueRange{args...});
    funCall.setCConv(LLVM::cconv::CConv::SPIR_FUNC);
    return funCall;
  }
};

class GenISA_Prefetching_2D
    : public GenISA<llvm::GenISAIntrinsic::ID::GenISA_LSC2DBlockPrefetch> {

public:
  using GenISA<llvm::GenISAIntrinsic::ID::GenISA_LSC2DBlockPrefetch>::GenISA;
};

class GenISA_Dpas
    : public GenISA<llvm::GenISAIntrinsic::ID::GenISA_sub_group_dpas> {

public:
  using GenISA<llvm::GenISAIntrinsic::ID::GenISA_sub_group_dpas>::GenISA;
};

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_VCINTRINSICHELPER_H
