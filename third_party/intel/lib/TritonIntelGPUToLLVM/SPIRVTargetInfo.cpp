//===- SPIRVTargetInfo.cpp - SPIRVTargetInfo implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SPIRVTargetInfo.h"
#include "Dialect/TritonIntelGPU/IR/Utils.h"
#include "SPIRVSubgroupOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::triton::intel {

namespace {

template <typename GroupOp>
Value createSPIRVGroupOp(RewriterBase &rewriter, Location loc, Type resultTy,
                         Value acc, spirv::GroupOperation spvGroupOp,
                         Value clusterSize = {}) {
  return GroupOp::create(rewriter, loc, resultTy, spirv::Scope::Subgroup,
                         spvGroupOp, acc, clusterSize);
}

/// Reduce over `numLanesToReduce` lanes, clustered when that is fewer than the
/// whole sub-group.
template <typename GroupOp>
Value createSPIRVGroupReduceOp(RewriterBase &rewriter, Location loc,
                               Type resultTy, Value acc,
                               unsigned numLanesToReduce, unsigned warpSize) {
  if (numLanesToReduce == warpSize)
    return createSPIRVGroupOp<GroupOp>(rewriter, loc, resultTy, acc,
                                       spirv::GroupOperation::Reduce);
  Value clusterSize =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                rewriter.getI32IntegerAttr(numLanesToReduce));
  return createSPIRVGroupOp<GroupOp>(rewriter, loc, resultTy, acc,
                                     spirv::GroupOperation::ClusteredReduce,
                                     clusterSize);
}

} // namespace

bool SPIRVTargetInfo::isSupportedWarpReduceOp(Operation *op,
                                              unsigned numLanesToReduce,
                                              unsigned warpSize) const {
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
             arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
             arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
             arith::XOrIOp>(op);
}

Value SPIRVTargetInfo::genWarpReduce(RewriterBase &rewriter, Location loc,
                                     Value acc, Operation *reduceOp,
                                     unsigned numLanesToReduce,
                                     unsigned warpSize) const {
  Type resultType = reduceOp->getResult(0).getType();
  // Use bit-equivalent logical operation for Boolean values.
  if (resultType.isInteger(1))
    return TypeSwitch<mlir::Operation *, Value>(reduceOp)
        .Case<arith::AddIOp, arith::MulIOp, arith::MaxSIOp, arith::MaxUIOp,
              arith::MinSIOp, arith::MinUIOp, arith::AndIOp, arith::OrIOp,
              arith::XOrIOp>([&](auto groupOp) {
          return createSPIRVGroupReduceOp<
              SPIRVLogicalGroupOpTy<decltype(groupOp)>>(
              rewriter, loc, resultType, acc, numLanesToReduce, warpSize);
        });
  return TypeSwitch<mlir::Operation *, Value>(reduceOp)
      .Case<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
            arith::MaxNumFOp, arith::MinNumFOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp>([&](auto groupOp) {
        return createSPIRVGroupReduceOp<SPIRVGroupOpTy<decltype(groupOp)>>(
            rewriter, loc, resultType, acc, numLanesToReduce, warpSize);
      });
}

FailureOr<Operation *>
SPIRVTargetInfo::matchSupportedWarpScanOp(triton::ScanOp op) const {
  return gpu::intel::matchEligibleSubgroupScan(op);
}

Value SPIRVTargetInfo::genWarpScan(RewriterBase &rewriter, Location loc,
                                   Value acc, Operation *combineOp,
                                   unsigned warpSize) const {
  // The op sets below must stay a subset of what `matchEligibleSubgroupScan`
  // admits: an unmatched `TypeSwitch` returns a null `Value`.
  Type resultType = combineOp->getResult(0).getType();
  if (resultType.isInteger(1))
    // The gate rejects `addi`/`maxsi`/`minsi` for `i1`.
    return TypeSwitch<mlir::Operation *, Value>(combineOp)
        .Case<arith::MulIOp, arith::MaxUIOp, arith::MinUIOp, arith::AndIOp,
              arith::OrIOp, arith::XOrIOp>([&](auto groupOp) {
          return createSPIRVGroupOp<SPIRVLogicalGroupOpTy<decltype(groupOp)>>(
              rewriter, loc, resultType, acc,
              spirv::GroupOperation::InclusiveScan);
        });
  return TypeSwitch<mlir::Operation *, Value>(combineOp)
      .Case<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
            arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
            arith::AndIOp, arith::OrIOp, arith::XOrIOp>([&](auto groupOp) {
        return createSPIRVGroupOp<SPIRVGroupOpTy<decltype(groupOp)>>(
            rewriter, loc, resultType, acc,
            spirv::GroupOperation::InclusiveScan);
      });
}

} // namespace mlir::triton::intel
