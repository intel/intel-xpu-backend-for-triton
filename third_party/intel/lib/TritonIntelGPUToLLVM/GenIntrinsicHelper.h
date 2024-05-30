//===- GenIntrinsicHelper.h - Gen intrinsic helper ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_VCINTRINSICHELPER_H
#define TRITON_VCINTRINSICHELPER_H

#include "TritonGENToLLVM/GenIntrinsics.h"
#include "Utility.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <mlir/Dialect/SPIRV/IR/SPIRVDialect.h>
#include <string>

namespace mlir {
namespace triton {
namespace intel {

mlir::LLVM::LLVMFuncOp
appendOrGetGenISADeclaration(OpBuilder &builder, llvm::GenISAIntrinsic::ID id,
                             ArrayRef<mlir::Type *> mlirTys);

class GenISA_WaveAll {
public:
  // Enum def copied from IGC.
  enum class WaveOps : unsigned int {
    SUM,
    PROD,
    UMIN,
    UMAX,
    IMIN,
    IMAX,
    OR,
    XOR,
    AND,
    FSUM,
    FPROD,
    FMIN,
    FMAX,
    UNDEF
  };

  explicit GenISA_WaveAll(OpBuilder &builder, Type retTy) : builder(builder) {
    // get GenISA intrinsic declaration.
    intrinsicDecl = appendOrGetGenISADeclaration(
        builder, llvm::GenISAIntrinsic::ID::GenISA_WaveAll, {&retTy});
  }

  template <typename... Args>
  Value operator()(OpBuilder &rewriter, Location loc, Args... args) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<LLVM::CallOp>(loc, retType, funName,
                                                 ValueRange{args...});
    return funCall.getResult();
  }

private:
  OpBuilder &builder;
  LLVM::LLVMFuncOp intrinsicDecl;
};

class GenISA_WaveCluster {
public:
  // Enum def copied from IGC.
  enum class WaveOps : unsigned int {
    SUM,
    PROD,
    UMIN,
    UMAX,
    IMIN,
    IMAX,
    OR,
    XOR,
    AND,
    FSUM,
    FPROD,
    FMIN,
    FMAX,
    UNDEF
  };

  explicit GenISA_WaveCluster(OpBuilder &builder, Type retTy)
      : builder(builder) {
    // get GenISA intrinsic declaration.
    intrinsicDecl = appendOrGetGenISADeclaration(
        builder, llvm::GenISAIntrinsic::ID::GenISA_WaveClustered, {&retTy});
  }

  template <typename... Args>
  Value operator()(OpBuilder &rewriter, Location loc, Args... args) {
    auto funName = intrinsicDecl.getName();
    auto retType = intrinsicDecl.getResultTypes();
    auto funCall = rewriter.create<LLVM::CallOp>(loc, retType, funName,
                                                 ValueRange{args...});
    return funCall.getResult();
  }

private:
  OpBuilder &builder;
  LLVM::LLVMFuncOp intrinsicDecl;
};

} // namespace intel
} // namespace triton
} // namespace mlir

#endif // TRITON_VCINTRINSICHELPER_H
