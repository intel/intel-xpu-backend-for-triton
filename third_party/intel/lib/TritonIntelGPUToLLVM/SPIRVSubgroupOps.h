//===- SPIRVSubgroupOps.h - Mapping for SPIR-V Reduction --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines mapping from operations in the 'arith' dialect to the
// corresponding SPIR-V Subgroup Reduction Operation.
//
//===----------------------------------------------------------------------===//

#ifndef TRITONINTELGPUTOLLVM_SPIRVSUBGROUPOPS_H
#define TRITONINTELGPUTOLLVM_SPIRVSUBGROUPOPS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;

namespace mlir::triton::intel {

template <typename OpTy> struct SPIRVArithmeticGroupOp {};

template <> struct SPIRVArithmeticGroupOp<arith::AddFOp> {
  using type = spirv::GroupNonUniformFAddOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::AddIOp> {
  using type = spirv::GroupNonUniformIAddOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MulFOp> {
  using type = spirv::GroupNonUniformFMulOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MulIOp> {
  using type = spirv::GroupNonUniformIMulOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MaxNumFOp> {
  using type = spirv::GroupNonUniformFMaxOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MinNumFOp> {
  using type = spirv::GroupNonUniformFMinOp;
};

template <typename OpTy>
using SPIRVArithmeticGroupOpTy = typename SPIRVArithmeticGroupOp<OpTy>::type;

template <typename OpTy> struct SPIRVBitwiseGroupOp {};

template <> struct SPIRVBitwiseGroupOp<arith::AndIOp> {
  using type = spirv::GroupNonUniformBitwiseAndOp;
};
template <> struct SPIRVBitwiseGroupOp<arith::OrIOp> {
  using type = spirv::GroupNonUniformBitwiseOrOp;
};
template <> struct SPIRVBitwiseGroupOp<arith::XOrIOp> {
  using type = spirv::GroupNonUniformBitwiseXorOp;
};

template <typename OpTy>
using SPIRVBitwiseGroupOpTy = typename SPIRVBitwiseGroupOp<OpTy>::type;

template <typename OpTy> struct SPIRVLogicalGroupOp {};

template <> struct SPIRVLogicalGroupOp<arith::AndIOp> {
  using type = spirv::GroupNonUniformLogicalAndOp;
};
template <> struct SPIRVLogicalGroupOp<arith::OrIOp> {
  using type = spirv::GroupNonUniformLogicalOrOp;
};
template <> struct SPIRVLogicalGroupOp<arith::XOrIOp> {
  using type = spirv::GroupNonUniformLogicalXorOp;
};

template <typename OpTy>
using SPIRVLogicalGroupOpTy = typename SPIRVLogicalGroupOp<OpTy>::type;

} // namespace mlir::triton::intel

#endif // TRITONINTELGPUTOLLVM_SPIRVSUBGROUPOPS_H
