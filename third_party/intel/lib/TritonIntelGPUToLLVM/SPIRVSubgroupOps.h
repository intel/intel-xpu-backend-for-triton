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
template <> struct SPIRVArithmeticGroupOp<arith::MaxSIOp> {
  using type = spirv::GroupNonUniformSMaxOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MaxUIOp> {
  using type = spirv::GroupNonUniformUMaxOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MinSIOp> {
  using type = spirv::GroupNonUniformSMinOp;
};
template <> struct SPIRVArithmeticGroupOp<arith::MinUIOp> {
  using type = spirv::GroupNonUniformUMinOp;
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

template <typename OpTy>
using is_spirv_arithmetic_group_op =
    llvm::is_one_of<OpTy, spirv::GroupNonUniformFAddOp,
                    spirv::GroupNonUniformIAddOp, spirv::GroupNonUniformFMulOp,
                    spirv::GroupNonUniformIMulOp, spirv::GroupNonUniformSMaxOp,
                    spirv::GroupNonUniformUMaxOp, spirv::GroupNonUniformSMinOp,
                    spirv::GroupNonUniformUMinOp, spirv::GroupNonUniformFMaxOp,
                    spirv::GroupNonUniformFMinOp>;

template <typename OpTy>
constexpr bool is_spirv_arithmetic_group_op_v =
    is_spirv_arithmetic_group_op<OpTy>::value;

template <typename OpTy>
using is_spirv_bitwise_group_op =
    llvm::is_one_of<OpTy, spirv::GroupNonUniformBitwiseAndOp,
                    spirv::GroupNonUniformBitwiseOrOp,
                    spirv::GroupNonUniformBitwiseXorOp>;

template <typename OpTy>
constexpr bool is_spirv_bitwise_group_op_v =
    is_spirv_bitwise_group_op<OpTy>::value;

template <typename OpTy>
using is_spirv_logical_group_op =
    llvm::is_one_of<OpTy, spirv::GroupNonUniformLogicalAndOp,
                    spirv::GroupNonUniformLogicalOrOp,
                    spirv::GroupNonUniformLogicalXorOp>;

template <typename OpTy>
constexpr bool is_spirv_logical_group_op_v =
    is_spirv_logical_group_op<OpTy>::value;

template <typename OpTy>
using is_spirv_group_op = std::disjunction<is_spirv_arithmetic_group_op<OpTy>,
                                           is_spirv_bitwise_group_op<OpTy>,
                                           is_spirv_logical_group_op<OpTy>>;

template <typename OpTy>
constexpr bool is_spirv_group_op_v = is_spirv_group_op<OpTy>::value;

} // namespace mlir::triton::intel

#endif // TRITONINTELGPUTOLLVM_SPIRVSUBGROUPOPS_H
