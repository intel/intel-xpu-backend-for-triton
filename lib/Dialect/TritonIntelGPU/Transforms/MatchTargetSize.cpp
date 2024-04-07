//===----- MatchTargetSize.cpp -------------------------------------- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// This file implements a transform pass to spilt operations so that each
/// operation would match the target IR size.
/// This pass should be run after tritonintelgpu-distribute-to-warps
/// This pass only support block pointer.
/// e.g.
/// 32x64xf32 = tt.dot 32x32xf16, 32x64xf16
/// will be split into 32 dots which match the configured DotSize
/// 8x16xf32 = tt.dot 8x16xf16, 16x16xf16
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"

class MatchTargetSizePass
    : public TritonIntelGPUMatchTargetSizeBase<MatchTargetSizePass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();
    dotAttrs.clear();
    sizePerAttrMap.clear();
    // set default supported dot size(m, n, k)
    if (!dotSize.hasValue()) {
      dotSize.addValue(8);
      dotSize.addValue(16);
      dotSize.addValue(16);
    }
    auto hasTensorType = [](Type type) {
      if (isa<RankedTensorType>(type))
        return true;
      else if (auto ptrType = dyn_cast<tt::PointerType>(type))
        if (isa<RankedTensorType>(ptrType.getPointeeType()))
          return true;
      return false;
    };
    m.walk([&](tt::DotOp dot) {
      auto type = cast<RankedTensorType>(dot.getResult().getType());
      dotAttrs.insert(type.getEncoding());
    });

    /// split op to match target IR size
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      SmallVector<Type> types(op->getOperandTypes().begin(),
                              op->getOperandTypes().end());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());
      types.append(resultTypes);
      if (!llvm::any_of(types, hasTensorType))
        ;
      else if (isa<scf::ForOp, scf::YieldOp>(op))
        ;
      else if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        recordRootSubSize(op->getResultTypes()[0]);
        transformArithConstantOp(cstOp);
      } else if (auto dot = dyn_cast<tt::DotOp>(op))
        transformDotOp(dot);
      else if (auto ptrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        recordRootSubSize(op->getResultTypes()[0]);
        transformMakeTensorPtrOp(ptrOp);
      }
      // arith,math,tt.advance,tt.load,tt.store,tt.prefetch
      else
        transformGenericOp(op);
      return WalkResult::advance();
    });

    /// canonicalize ops(remove redundant tt.extract, tt.glue)
    RewritePatternSet patterns(ctx);
    patterns.add<ScfPattern>(ctx);
    patterns.add<ExtractPattern>(ctx);
    // FIXME: upstream to arith dialect
    patterns.add<ArithRemPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
      signalPassFailure();
  }

private:
  DenseMap<Attribute, SmallVector<int64_t>> sizePerAttrMap;
  DenseSet<Attribute> dotAttrs;
  void recordRootSubSize(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      Attribute layout = tensorType.getEncoding();
      if (sizePerAttrMap.count(layout) == 0)
        sizePerAttrMap[layout] = getSubOpSize(tensorType);
    } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      recordRootSubSize(ptrType.getPointeeType());
    }
  }
  SmallVector<int64_t> getSubOpSize(RankedTensorType type) {
    int64_t mStep = dotSize[0];
    int64_t nStep = dotSize[1];
    int64_t kStep = dotSize[2];
    Attribute layout = type.getEncoding();
    int64_t colLimit = 0;
    if (dotAttrs.count(layout)) {
      SmallVector<int64_t> subSize{mStep, nStep};
      return subSize;
      // FIXME: 2 * subgroupSize
    } else if (auto warpAttr = dyn_cast<ttgi::WarpEncodingAttr>(layout)) {
      colLimit = 32;
    } else if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
      colLimit = 32;
    }
    ArrayRef<int64_t> shape = type.getShape();
    SmallVector<int64_t> subSize(shape.size());
    unsigned sizeInByte = type.getElementTypeBitWidth() / 8;
    if (shape.size() == 2) {
      subSize[1] = (shape[1] > colLimit) ? colLimit : shape[1];
      // all load/store size max is 512 DW
      auto max = 512 * 4 / sizeInByte / subSize[1];
      subSize[0] = std::min(max, shape[0]);
    } else if (shape.size() == 1) {
      int64_t max = 512 * 4 / sizeInByte;
      subSize[0] = std::min(max, shape[0]);
    }
    return subSize;
  }

  // return [shape, subType, subSize]
  std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
  getSubTypeAndShape(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      SmallVector<int64_t> shape = llvm::to_vector(tensorType.getShape());
      Attribute attr = tensorType.getEncoding();
      SmallVector<int64_t> subSize = sizePerAttrMap[attr];
      auto subType = RankedTensorType::get(
          subSize, tensorType.getElementType() /*no encoding*/);
      return std::make_tuple(shape, subType, subSize);
    } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      Type pointeeType = ptrType.getPointeeType();
      auto [shape, subType, subSize] = getSubTypeAndShape(pointeeType);
      auto newType = tt::PointerType::get(subType, ptrType.getAddressSpace());
      return std::make_tuple(shape, newType, subSize);
    }
    return {{0}, type, {0}};
  }

  void transformMakeTensorPtrOp(tt::MakeTensorPtrOp op) {
    Type type = op.getType();
    auto [shape, subType, subSize] = getSubTypeAndShape(type);
    // // early return
    // if (shape == subSize)
    //   return;
    unsigned dim = shape.size();
    OpBuilder b(op);
    Location loc = op.getLoc();
    unsigned idx = 0;
    SmallVector<Value> subOps;
    for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (unsigned j = 0; j < shape[0]; j += subSize[0]) {
          auto offsets = op.getOffsets();
          // newOffsets = offsets += [j, i]
          SmallVector<Value> newOffsets(2);
          if (j == 0)
            newOffsets[0] = offsets[0];
          else {
            auto step = b.create<arith::ConstantIntOp>(loc, j, 32);
            newOffsets[0] = b.create<arith::AddIOp>(loc, offsets[0], step);
          }
          if (i == 0)
            newOffsets[1] = offsets[1];
          else {
            auto step = b.create<arith::ConstantIntOp>(loc, i, 32);
            newOffsets[1] = b.create<arith::AddIOp>(loc, offsets[1], step);
          }
          SmallVector<int32_t> subShape;
          for (auto sub : subSize)
            subShape.push_back(sub);
          auto subOp = b.create<tt::MakeTensorPtrOp>(
              loc, op.getBase(), op.getShape(), op.getStrides(), newOffsets,
              subShape, op.getOrder());
          subOps.push_back(subOp);
        }
      }
    }
    auto glue = b.create<ttgi::GlueOp>(loc, type, subOps);
    op->replaceAllUsesWith(glue->getResults());
    op->erase();
    return;
  }

  void transformArithConstantOp(arith::ConstantOp op) {
    auto type = cast<RankedTensorType>(op.getResult().getType());
    auto [shape, subType, subSize] = getSubTypeAndShape(type);
    // // early return
    // if (shape == subSize)
    //   return;
    unsigned dim = shape.size();
    OpBuilder b(op);
    Location loc = op.getLoc();
    unsigned idx = 0;
    auto value = cast<DenseElementsAttr>(op.getValue());
    value = value.resizeSplat(subType.cast<ShapedType>());
    SmallVector<Value> subOps;
    for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (unsigned j = 0; j < shape[0]; j += subSize[0]) {
          auto subOp = b.create<arith::ConstantOp>(loc, subType, value);
          subOps.push_back(subOp);
        }
      }
    }
    auto glue = b.create<ttgi::GlueOp>(loc, type, subOps);
    op->replaceAllUsesWith(glue->getResults());
    op->erase();
    return;
  }

  void transformGenericOp(Operation *op) {
    unsigned numResults = op->getResults().size();
    unsigned dotIdx = 2;
    Type type;
    // prefetch/store
    if (numResults == 0)
      type = op->getOperand(0).getType();
    // arith/math/advanceOp/loadOp
    else {
      assert(numResults == 1 && "op should have 1 result");
      type = op->getResultTypes()[0];
      // mark tt.load for dot A/B
      auto tensorType = dyn_cast<RankedTensorType>(type);
      if (dyn_cast<tt::LoadOp>(op) && tensorType) {
        Attribute layout = tensorType.getEncoding();
        if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
          dotIdx = dotAttr.getOpIdx();
        }
      }
    }
    auto [shape, subType, subSize] = getSubTypeAndShape(type);
    unsigned dim = shape.size();
    OpBuilder b(op);
    Location loc = op->getLoc();
    unsigned idx = 0;
    SmallVector<Value> subOps;
    for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (unsigned j = 0; j < shape[0]; j += subSize[0]) {
          SmallVector<Value> newOperands;
          llvm::transform(op->getOperands(), std::back_inserter(newOperands),
                          [&](Value operand) {
                            Type type = operand.getType();
                            if (isa<tt::PointerType, RankedTensorType>(type)) {
                              Type subOpndType =
                                  std::get<1>(getSubTypeAndShape(type));
                              Value newOp = b.create<ttgi::ExtractOp>(
                                  loc, subOpndType, operand, idx);
                              return newOp;
                            } else
                              return operand;
                          });
          Operation *subOp;
          if (numResults == 0)
            subOp = b.create(loc, op->getName().getIdentifier(), newOperands,
                             {}, op->getAttrs());
          else {
            subOp = b.create(loc, op->getName().getIdentifier(), newOperands,
                             subType, op->getAttrs());
            if (dotIdx < 2)
              subOp->setAttr("DotIdx",
                             b.getIntegerAttr(b.getI32Type(), dotIdx));
            subOps.push_back(subOp->getResults()[0]);
          }
          idx++;
        }
      }
    }
    if (numResults == 1) {
      auto glue = b.create<ttgi::GlueOp>(loc, type, subOps);
      op->replaceAllUsesWith(glue);
    }
    op->erase();
  }

  void transformDotOp(tt::DotOp dot) {
    assert(dotSize.size() == 3 && "dot-size should have m, n ,k");
    auto aType = dot.getA().getType().cast<RankedTensorType>();
    auto bType = dot.getB().getType().cast<RankedTensorType>();
    auto cType = dot.getC().getType().cast<RankedTensorType>();
    ArrayRef<int64_t> aShape = aType.getShape();
    ArrayRef<int64_t> bShape = bType.getShape();
    ArrayRef<int64_t> cShape = cType.getShape();
    int64_t m = aShape[0];
    int64_t n = bShape[1];
    int64_t k = aShape[1];
    int64_t mStep = dotSize[0];
    int64_t nStep = dotSize[1];
    int64_t kStep = dotSize[2];
    OpBuilder b(dot);
    Location loc = dot.getLoc();
    auto getSubDotVal = [&](Value val, int64_t mm, int64_t kk, int64_t mStep,
                            int64_t kStep) {
      auto [shape, subType, subSize] = getSubTypeAndShape(val.getType());
      unsigned subIdx =
          (kk / subSize[1]) * (shape[0] / subSize[0]) + mm / subSize[0];
      Value subVal = b.create<ttgi::ExtractOp>(loc, subType, val, subIdx);
      auto subDotType = RankedTensorType::get(
          {mStep, kStep},
          val.getType().cast<RankedTensorType>().getElementType());
      unsigned subDotIdx = ((kk % subSize[1]) / kStep) * (subSize[0] / mStep) +
                           (mm % subSize[0]) / mStep;
      Value subDotVal =
          b.create<ttgi::ExtractOp>(loc, subDotType, subVal, subDotIdx);
      return subDotVal;
    };

    auto [shape, subType, subSize] = getSubTypeAndShape(cType);
    SmallVector<Value> subCs;
    for (unsigned nn = 0; nn < n; nn += nStep) {
      for (unsigned mm = 0; mm < m; mm += mStep) {
        Value subDotC = getSubDotVal(dot.getC(), mm, nn, mStep, nStep);
        for (unsigned kk = 0; kk < k; kk += kStep) {
          auto subDotA = getSubDotVal(dot.getA(), mm, kk, mStep, kStep);
          auto subDotB = getSubDotVal(dot.getB(), kk, nn, kStep, nStep);
          subDotC = b.create<tt::DotOp>(loc, subDotA, subDotB, subDotC,
                                        dot.getInputPrecisionAttr(),
                                        dot.getMaxNumImpreciseAccAttr());
        }
        subCs.push_back(subDotC);
      }
    }
    auto newC = b.create<ttgi::GlueOp>(loc, dot.getType(), subCs);
    dot->replaceAllUsesWith(newC->getResults());
    dot->erase();
  }

  class ArithRemPattern : public OpRewritePattern<arith::RemSIOp> {
  public:
    using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(arith::RemSIOp op,
                                  PatternRewriter &rewriter) const final {
      Location loc = op.getLoc();
      Type type = op.getType();
      Value rhs = op.getRhs();
      APInt value;
      if (!matchPattern(rhs, m_ConstantInt(&value)))
        return failure();
      if (value.isOne()) {
        IntegerAttr attr = rewriter.getIntegerAttr(type, 0);
        auto zero = rewriter.create<arith::ConstantOp>(loc, attr);
        rewriter.replaceOp(op, zero);
        return success();
      } else if (value.popcount() == 1) {
        IntegerAttr attr =
            rewriter.getIntegerAttr(type, value.getSExtValue() - 1);
        auto mask = rewriter.create<arith::ConstantOp>(loc, attr);
        auto result = rewriter.create<arith::AndIOp>(loc, op.getLhs(), mask);
        rewriter.replaceOp(op, result);
        return success();
      }
      return failure();
    }
  };

  // assume that no sideEffect op in between
  class ExtractPattern : public OpRewritePattern<ttgi::ExtractOp> {
  public:
    using OpRewritePattern<ttgi::ExtractOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(ttgi::ExtractOp op,
                                  PatternRewriter &rewriter) const final {
      Value base = op.getBase();
      if (Operation *def = base.getDefiningOp()) {
        if (auto glue = dyn_cast<ttgi::GlueOp>(def)) {
          Value sub = glue->getOperand(op.getIdx());
          rewriter.replaceOp(op, sub);
          return success();
        }
      }
      if (base.getType() == op.getType() && op.getIdx() == 0) {
        rewriter.replaceOp(op, base);
        return success();
      }
      return failure();
    }
  };
  class ScfPattern : public OpRewritePattern<scf::ForOp> {
  public:
    using OpRewritePattern<scf::ForOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(scf::ForOp op,
                                  PatternRewriter &rewriter) const final {
      SmallVector<Operation *> deleteList;
      SmallVector<Value> newInits;
      DenseMap<Value, int> userIndexMap;
      unsigned idx = 0;
      for (auto [arg, init] :
           llvm::zip(op.getRegionIterArgs(), op.getInits())) {
        auto glue = dyn_cast<ttgi::GlueOp>(init.getDefiningOp());
        if (!glue) {
          newInits.push_back(init);
          userIndexMap[arg] = idx;
          idx++;
          continue;
        }
        unsigned numSplit = glue->getOperands().size();
        for (unsigned i = 0; i < numSplit; i++) {
          newInits.push_back(glue->getOperand(i));
        }
        for (auto *user : arg.getUsers()) {
          auto extract = dyn_cast<ttgi::ExtractOp>(user);
          if (extract) {
            userIndexMap[extract] = idx + extract.getIdx();
            deleteList.push_back(extract.getOperation());
          }
        }
        idx += numSplit;
      }
      if (newInits.size() == op.getInits().size())
        return failure();
      auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                               op.getUpperBound(), op.getStep(),
                                               newInits);

      for (auto [user, idx] : userIndexMap)
        user.replaceAllUsesWith(newOp.getRegionIterArgs()[idx]);
      op.getInductionVar().replaceAllUsesWith(newOp.getInductionVar());
      // splice operations
      Block *body = newOp.getBody();
      body->getOperations().splice(body->begin(),
                                   op.getBody()->getOperations());
      // yield op
      auto yield = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> newValues;
      for (auto result : yield.getResults()) {
        Operation *def = result.getDefiningOp();
        if (def) {
          if (auto glue = dyn_cast<ttgi::GlueOp>(def)) {
            newValues.append(glue->getOperands().begin(),
                             glue->getOperands().end());
          } else {
            newValues.push_back(result);
          }
        }
      }
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(yield);
        rewriter.create<scf::YieldOp>(yield.getLoc(), newValues);
        rewriter.eraseOp(yield);
      }

      // replace results
      userIndexMap.clear();
      idx = 0;
      for (auto [result, init] : llvm::zip(op.getResults(), op.getInits())) {
        auto glue = dyn_cast<ttgi::GlueOp>(init.getDefiningOp());
        if (!glue) {
          userIndexMap[result] = idx;
          idx++;
          continue;
        }
        for (auto user : result.getUsers()) {
          auto extract = dyn_cast<ttgi::ExtractOp>(user);
          if (extract) {
            userIndexMap[extract] = idx + extract.getIdx();
            deleteList.push_back(extract.getOperation());
          }
        }
        idx += glue->getOperands().size();
      }
      for (auto [user, idx] : userIndexMap)
        user.replaceAllUsesWith(newOp.getResults()[idx]);
      for (auto op : deleteList)
        rewriter.eraseOp(op);
      rewriter.eraseOp(op);
      return success();
    }
  };
};

std::unique_ptr<Pass> mlir::triton::gpu::intel::createMatchTargetSizePass() {
  return std::make_unique<MatchTargetSizePass>();
}
