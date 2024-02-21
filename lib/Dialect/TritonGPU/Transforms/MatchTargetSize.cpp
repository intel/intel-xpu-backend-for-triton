#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUMatchTargetSizePass
    : public TritonGPUMatchTargetSizeBase<TritonGPUMatchTargetSizePass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp m = getOperation();
    sizePerAttrMap.clear();
    if (!dotSize.hasValue()) {
      dotSize.addValue(8);
      dotSize.addValue(16);
      dotSize.addValue(16);
    }
    /// split op to match target llvm/spirv size
    // try to use callback to handle different cases
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      SmallVector<Type> types(op->getOperandTypes().begin(),
                              op->getOperandTypes().end());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());
      types.append(resultTypes);
      if (!llvm::any_of(types, [&](Type type) {
            if (isa<RankedTensorType>(type)) {
              return true;
            } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
              auto pointeeType = ptrType.getPointeeType();
              return isa<RankedTensorType>(pointeeType) ? true : false;
            } else {
              return false;
            }
          }))
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

    m.dump();
    /// canonicalize ops(remove redundant tt.extract, tt.glue)
    RewritePatternSet patterns(ctx);
    patterns.add<ScfPattern>(ctx);
    patterns.add<ExtractPattern>(ctx);
    patterns.add<ArithRemPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
      signalPassFailure();
  }

private:
  DenseMap<Attribute, SmallVector<int64_t>> sizePerAttrMap;
  void recordRootSubSize(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      auto layout = tensorType.getEncoding();
      if (sizePerAttrMap.count(layout) == 0)
        sizePerAttrMap[layout] = getSubOpSize(tensorType);
    } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      recordRootSubSize(ptrType.getPointeeType());
    }
  }
  // assume
  // all lsc size max is 256 DW
  // lsc 2d limit is 32
  //  bool isLsc = false)
  SmallVector<int64_t> getSubOpSize(RankedTensorType type) {
    auto mStep = dotSize[0];
    auto nStep = dotSize[1];
    auto kStep = dotSize[2];
    auto layout = type.getEncoding();
    int64_t colLimit = 0;
    if (auto warpAttr = dyn_cast<ttg::WarpEncodingAttr>(layout)) {
      colLimit = 32;
      colLimit = 16;
      // if (warpAttr.getIsDotC())
      if (warpAttr.getSizePerThread()[1] == 64)
        colLimit = nStep;
    } else if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
      colLimit = dotAttr.getOpIdx() == 0 ? kStep : nStep;
    }
    auto shape = type.getShape();
    SmallVector<int64_t> subSize(shape.size());
    auto sizeInByte = type.getElementTypeBitWidth() / 8;
    if (shape.size() == 2) {
      subSize[1] = (shape[1] > colLimit) ? colLimit : shape[1];
      auto max = 256 * 4 / sizeInByte / subSize[1];
      subSize[0] = std::min(max, shape[0]);
    } else if (shape.size() == 1) {
      int64_t max = 256 * 4 / sizeInByte;
      subSize[0] = std::min(max, shape[0]);
    }
    return subSize;
  }

  std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
  getSubTypeAndShape(Type type) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
      auto shape = llvm::to_vector(tensorType.getShape());
      auto attr = tensorType.getEncoding();
      auto subSize = sizePerAttrMap[attr];
      auto subType = RankedTensorType::get(
          subSize, tensorType.getElementType() /*no encoding*/);
      return std::make_tuple(shape, subType, subSize);
    } else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      auto pointeeType = ptrType.getPointeeType();
      auto [shape, subType, subSize] = getSubTypeAndShape(pointeeType);
      auto newType = tt::PointerType::get(subType, ptrType.getAddressSpace());
      return std::make_tuple(shape, newType, subSize);
    }
    return {{0}, type, {0}};
  }

  void transformMakeTensorPtrOp(tt::MakeTensorPtrOp op) {
    auto type = op.getType();
    auto [shape, subType, subSize] = getSubTypeAndShape(type);
    // // early return
    // if (shape == subSize)
    //   return;
    auto dim = shape.size();
    OpBuilder b(op);
    auto loc = op.getLoc();
    auto idx = 0;
    SmallVector<Value> subOps;
    for (auto i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (auto j = 0; j < shape[0]; j += subSize[0]) {
          auto offsets = op.getOffsets();
          // auto newOffsets = offsets += [i, j]
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
    auto glue = b.create<tt::GlueOp>(loc, type, subOps);
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
    auto dim = shape.size();
    OpBuilder b(op);
    auto loc = op.getLoc();
    auto idx = 0;
    // these 2 lines are specific
    auto value = cast<DenseElementsAttr>(op.getValue());
    value = value.resizeSplat(subType.cast<ShapedType>());
    SmallVector<Value> subOps;
    for (auto i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (auto j = 0; j < shape[0]; j += subSize[0]) {
          auto subOp = b.create<arith::ConstantOp>(loc, subType, value);
          subOps.push_back(subOp);
        }
      }
    }
    auto glue = b.create<tt::GlueOp>(loc, type, subOps);
    op->replaceAllUsesWith(glue->getResults());
    op->erase();
    return;
  }

  void transformGenericOp(Operation *op) {
    auto numResults = op->getResults().size();
    Type type;
    // prefetch/store
    if (numResults == 0)
      type = op->getOperand(0).getType();
    // arith/math/advanceOp
    else {
      assert(numResults == 1);
      type = op->getResultTypes()[0];
    }
    auto [shape, subType, subSize] = getSubTypeAndShape(type);
    // // early return
    // if (shape == subSize)
    //   return;
    auto dim = shape.size();
    OpBuilder b(op);
    auto loc = op->getLoc();
    auto idx = 0;
    SmallVector<Value> subOps;
    for (auto i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
      if (dim == 2) {
        for (auto j = 0; j < shape[0]; j += subSize[0]) {
          SmallVector<Value> newOperands;
          llvm::transform(op->getOperands(), std::back_inserter(newOperands),
                          [&](Value operand) {
                            auto type = operand.getType();
                            if (isa<tt::PointerType, RankedTensorType>(type)) {
                              auto subOpndType =
                                  std::get<1>(getSubTypeAndShape(type));
                              Value newOp = b.create<tt::ExtractOp>(
                                  loc, subOpndType, operand, idx);
                              return newOp;
                            } else
                              return operand;
                          });
          // auto subOp = b.create<op->getName().getTypeID()>(loc, newOperands,
          //                                                  op->getAttrs());
          // subOps.push_back(subOp);
          Operation *subOp;
          if (numResults == 0)
            subOp = b.create(loc, op->getName().getIdentifier(), newOperands,
                             {}, op->getAttrs());
          else {
            subOp = b.create(loc, op->getName().getIdentifier(), newOperands,
                             subType, op->getAttrs());
            subOps.push_back(subOp->getResults()[0]);
          }
          idx++;
        }
      }
    }
    if (numResults == 1) {
      auto glue = b.create<tt::GlueOp>(loc, type, subOps);
      op->replaceAllUsesWith(glue);
    }
    op->erase();
  }

  // for dot split k, n to make the inner register contiguous
  void transformDotOp(tt::DotOp dot) {
    assert(dotSize.size() == 3 && "target-size should have m, n ,k");
    auto aType = dot.getA().getType().cast<RankedTensorType>();
    auto bType = dot.getB().getType().cast<RankedTensorType>();
    auto cType = dot.getC().getType().cast<RankedTensorType>();
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    auto cShape = cType.getShape();
    auto m = aShape[0];
    auto n = bShape[1];
    auto k = aShape[1];
    auto mStep = dotSize[0];
    auto nStep = dotSize[1];
    auto kStep = dotSize[2];
    OpBuilder b(dot);
    auto loc = dot.getLoc();
    auto packValues = [&](ArrayRef<int64_t> values) {
      SmallVector<OpFoldResult> newValues = llvm::to_vector<4>(
          llvm::map_range(values, [&](int64_t v) -> OpFoldResult {
            return b.getI64IntegerAttr(v);
          }));
      return newValues;
    };
    auto getSubDotVal = [&](Value val, int64_t mm, int64_t kk, int64_t mStep,
                            int64_t kStep) {
      auto [shape, subType, subSize] = getSubTypeAndShape(val.getType());
      auto subIdx =
          (kk / subSize[1]) * (shape[0] / subSize[0]) + mm / subSize[0];
      auto subVal = b.create<tt::ExtractOp>(loc, subType, val, subIdx);
      auto subDotType = RankedTensorType::get(
          {mStep, kStep},
          val.getType().cast<RankedTensorType>().getElementType());
      auto subDotIdx = ((kk % subSize[1]) / kStep) * (subSize[0] / mStep) +
                       (mm % subSize[0]) / mStep;
      auto subDotVal =
          b.create<tt::ExtractOp>(loc, subDotType, subVal, subDotIdx);
      return subDotVal;
    };

    // n first, so that we can use larger store
    auto [shape, subType, subSize] = getSubTypeAndShape(cType);
    SmallVector<Value> subCs;
    SmallVector<Value> subOps;
    for (auto nn = 0; nn < n; nn += nStep) {
      for (auto mm = 0; mm < m; mm += mStep) {
        if (mm % subSize[0] == 0) {
          subOps.clear();
        }
        Value subDotC = getSubDotVal(dot.getC(), mm, nn, mStep, nStep);
        for (auto kk = 0; kk < k; kk += kStep) {
          auto subDotA = getSubDotVal(dot.getA(), mm, kk, mStep, kStep);
          auto subDotB = getSubDotVal(dot.getB(), kk, nn, kStep, nStep);
          subDotC = b.create<tt::DotOp>(loc, subDotA, subDotB, subDotC,
                                        dot.getAllowTF32Attr(),
                                        dot.getMaxNumImpreciseAccAttr());
        }
        subOps.push_back(subDotC);
        if ((mm % subSize[0]) == (subSize[0] - mStep)) {
          Value glue = b.create<tt::GlueOp>(loc, subType, subOps);
          subCs.push_back(glue);
        }
      }
    }
    auto newC = b.create<tt::GlueOp>(loc, dot.getType(), subCs);
    dot->replaceAllUsesWith(newC->getResults());
    dot->erase();
  }

  class ArithRemPattern : public OpRewritePattern<arith::RemSIOp> {
  public:
    using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(arith::RemSIOp op,
                                  PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      auto type = op.getType();
      auto rhs = op.getRhs();
      APInt value;
      if (!matchPattern(rhs, m_ConstantInt(&value)))
        return failure();
      if (value.isOne()) {
        auto attr = rewriter.getIntegerAttr(type, 0);
        auto zero = rewriter.create<arith::ConstantOp>(loc, attr);
        rewriter.replaceOp(op, zero);
        return success();
      } else if (value.popcount() == 1) {
        auto attr = rewriter.getIntegerAttr(type, value.getSExtValue() - 1);
        auto mask = rewriter.create<arith::ConstantOp>(loc, attr);
        auto result = rewriter.create<arith::AndIOp>(loc, op.getLhs(), mask);
        rewriter.replaceOp(op, result);
        return success();
      }
      return failure();
    }
  };

  // assume that no sideEffect op in between
  class ExtractPattern : public OpRewritePattern<tt::ExtractOp> {
  public:
    using OpRewritePattern<tt::ExtractOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tt::ExtractOp op,
                                  PatternRewriter &rewriter) const final {
      auto base = op.getBase();
      if (auto def = base.getDefiningOp()) {
        if (auto glue = dyn_cast<tt::GlueOp>(def)) {
          auto sub = glue->getOperand(op.getIdx());
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
      auto idx = 0;
      for (auto [arg, init] :
           llvm::zip(op.getRegionIterArgs(), op.getInits())) {
        auto glue = dyn_cast<tt::GlueOp>(init.getDefiningOp());
        if (!glue) {
          newInits.push_back(init);
          userIndexMap[arg] = idx;
          idx++;
          continue;
        }
        auto numSplit = glue->getOperands().size();
        for (auto i = 0; i < numSplit; i++) {
          newInits.push_back(glue->getOperand(i));
        }
        for (auto user : arg.getUsers()) {
          auto extract = dyn_cast<tt::ExtractOp>(user);
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
      auto *body = newOp.getBody();
      body->getOperations().splice(body->begin(),
                                   op.getBody()->getOperations());
      // yield op
      auto yield = cast<scf::YieldOp>(body->getTerminator());
      SmallVector<Value> newValues;
      for (auto result : yield.getResults()) {
        auto def = result.getDefiningOp();
        if (def) {
          if (auto glue = dyn_cast<tt::GlueOp>(def)) {
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

      // replaceResults
      userIndexMap.clear();
      idx = 0;
      for (auto [result, init] : llvm::zip(op.getResults(), op.getInits())) {
        auto glue = dyn_cast<tt::GlueOp>(init.getDefiningOp());
        if (!glue) {
          userIndexMap[result] = idx;
          idx++;
          continue;
        }
        for (auto user : result.getUsers()) {
          auto extract = dyn_cast<tt::ExtractOp>(user);
          if (extract) {
            userIndexMap[extract] = idx + extract.getIdx();
            deleteList.push_back(extract.getOperation());
          }
        }
        idx += glue->getOperands().size();
      }
      for (auto [user, idx] : userIndexMap)
        user.replaceAllUsesWith(newOp.getResults()[idx]);
      // rewriter.replaceAllUsesWith()
      for (auto op : deleteList)
        rewriter.eraseOp(op);
      rewriter.eraseOp(op);
      return success();
    }
  };
};

std::unique_ptr<Pass> mlir::triton::gpu::createTritonGPUMatchTargetSizePass() {
  return std::make_unique<TritonGPUMatchTargetSizePass>();
}
