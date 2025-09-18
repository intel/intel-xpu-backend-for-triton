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

template <typename OpTy> struct SPIRVGroupOp {};

template <> struct SPIRVGroupOp<arith::AddFOp> {
  using type = spirv::GroupNonUniformFAddOp;
};
template <> struct SPIRVGroupOp<arith::AddIOp> {
  using type = spirv::GroupNonUniformIAddOp;
};
template <> struct SPIRVGroupOp<arith::MulFOp> {
  using type = spirv::GroupNonUniformFMulOp;
};
template <> struct SPIRVGroupOp<arith::MulIOp> {
  using type = spirv::GroupNonUniformIMulOp;
};
template <> struct SPIRVGroupOp<arith::MaxSIOp> {
  using type = spirv::GroupNonUniformSMaxOp;
};
template <> struct SPIRVGroupOp<arith::MaxUIOp> {
  using type = spirv::GroupNonUniformUMaxOp;
};
template <> struct SPIRVGroupOp<arith::MinSIOp> {
  using type = spirv::GroupNonUniformSMinOp;
};
template <> struct SPIRVGroupOp<arith::MinUIOp> {
  using type = spirv::GroupNonUniformUMinOp;
};
template <> struct SPIRVGroupOp<arith::MinNumFOp> {
  using type = spirv::GroupNonUniformFMinOp;
};
template <> struct SPIRVGroupOp<arith::AndIOp> {
  using type = spirv::GroupNonUniformBitwiseAndOp;
};
template <> struct SPIRVGroupOp<arith::OrIOp> {
  using type = spirv::GroupNonUniformBitwiseOrOp;
};
template <> struct SPIRVGroupOp<arith::XOrIOp> {
  using type = spirv::GroupNonUniformBitwiseXorOp;
};

template <typename OpTy>
using SPIRVGroupOpTy = typename SPIRVGroupOp<OpTy>::type;

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
template <> struct SPIRVLogicalGroupOp<arith::AddIOp> {
  using type = spirv::GroupNonUniformLogicalOrOp;
};
template <> struct SPIRVLogicalGroupOp<arith::MulIOp> {
  using type = spirv::GroupNonUniformLogicalAndOp;
};
template <> struct SPIRVLogicalGroupOp<arith::MaxUIOp> {
  using type = spirv::GroupNonUniformLogicalOrOp;
};
template <> struct SPIRVLogicalGroupOp<arith::MaxSIOp> {
  using type = spirv::GroupNonUniformLogicalOrOp;
};
template <> struct SPIRVLogicalGroupOp<arith::MinUIOp> {
  using type = spirv::GroupNonUniformLogicalAndOp;
};
template <> struct SPIRVLogicalGroupOp<arith::MinSIOp> {
  using type = spirv::GroupNonUniformLogicalAndOp;
};

template <typename OpTy>
using SPIRVLogicalGroupOpTy = typename SPIRVLogicalGroupOp<OpTy>::type;

} // namespace mlir::triton::intel

#endif // TRITONINTELGPUTOLLVM_SPIRVSUBGROUPOPS_H
