//===- SPIRVTargetInfo.cpp - SPIRVTargetInfo implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVTargetInfo.h"
#include "SPIRVSubgroupOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::triton::intel {

namespace {

template <typename GroupOp>
Value createSPIRVGroupOp(RewriterBase &rewriter, Location loc, Type resultTy,
                         Value acc, unsigned activeLanes,
                         unsigned warpSize) {
  auto spvGroupOp = spirv::GroupOperation::Reduce;
  Value clusterSize;
  if (activeLanes != (warpSize - 1)) {
    spvGroupOp = spirv::GroupOperation::ClusteredReduce;
    clusterSize =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(activeLanes));
  }

  return GroupOp::create(rewriter, loc, resultTy, spirv::Scope::Subgroup,
                         spvGroupOp, acc, clusterSize);
}

} // namespace

bool SPIRVTargetInfo::isSupportedWarpReduceOp(Operation *op,
                                              unsigned warpSize) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
             arith::XOrIOp>(op);
}

Value SPIRVTargetInfo::genWarpReduce(RewriterBase &rewriter, Location loc,
                                     Value acc, Operation *reduceOp,
                                     unsigned activeLanes,
                                     unsigned warpSize) const {
  Type resultType = reduceOp->getResult(0).getType();
  // Use bit-equivalent logical operation for Boolean values.
  if (resultType.isInteger(1))
    return TypeSwitch<mlir::Operation *, Value>(reduceOp)
        .Case<arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MaxUIOp,
              arith::MinSIOp, arith::MinUIOp, arith::AndIOp, arith::OrIOp,
              arith::XOrIOp>([&](auto groupOp) {
          return createSPIRVGroupOp<SPIRVLogicalGroupOpTy<decltype(groupOp)>>(
              rewriter, loc, resultType, acc, activeLanes, warpSize);
        });
  return TypeSwitch<mlir::Operation *, Value>(reduceOp)
      .Case<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp>([&](auto groupOp) {
        return createSPIRVGroupOp<SPIRVGroupOpTy<decltype(groupOp)>>(
            rewriter, loc, resultType, acc, activeLanes, warpSize);
      });
}

} // namespace mlir::triton::intel
