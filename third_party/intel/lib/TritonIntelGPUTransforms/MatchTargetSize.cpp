//===- MatchTargetSize.cpp ----------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass designed to split various triton operations so
/// that each one matches the native size supported by the target architecture.
///
/// Notes:
///   - only blocked pointers are supported
///   - it is expected that the 'tritonintelgpu-distribute-to-warps' pass is run
///     before this pass
///
/// For example, the following `tt.dot` operation:
///
///   %0 = tt.dot %a, %b : tensor<32x32xf16>, tensor<32x64xf16>
///      -> tensor<32x64xf32>
///
/// is splits it into 32 `tt.dot` operations with tensor sizes matching the
/// target architecture 'dot' size of <8x16xf32>:
///
///   %tile_a0 = triton_intel_gpu.extract %a[0] : tensor<32x32xf16>
///            -> tensor<8x16xf16>
///   %tile_b0 = triton_intel_gpu.extract %b[0] : tensor<32x32xf16>
///            -> tensor<16x16xf16>
///   %dot_0 = tt.dot %tile_a0, %tile_b0 : tensor<8x16xf16>, tensor<16x16xf16>
///          -> tensor<8x16xf32>
///   %tile_a4 = triton_intel_gpu.extract %a[4] : tensor<32x32xf16>
///            -> tensor<8x16xf16>
///   %tile_b1 = triton_intel_gpu.extract %b[1] : tensor<32x32xf16>
///            -> tensor<16x16xf16>
///   %dot_1 = tt.dot %tile_a4, %tile_b1 : tensor<8x16xf16>, tensor<16x16xf16>
///          -> tensor<8x16xf32>
///   ...
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUMATCHTARGETSIZE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-match-target-size"

namespace {

// Encode the native operation sizes supported by the target architecture.
class TargetArchNativeSizes {
public:
  struct DotShape {
    DotShape() = default;
    DotShape(unsigned m, unsigned n, unsigned k) : m(m), n(n), k(k) {
      assert(m != 0 && n != 0 && k != 0 && "expecting valid shape");
    }

    unsigned m = 0;
    unsigned n = 0;
    unsigned k = 0;
  };

  struct BlockMemShape {
    BlockMemShape() = default;
    BlockMemShape(unsigned rowsA, unsigned columnsA, unsigned rowsB,
                  unsigned columnsB)
        : rowsA(rowsA), columnsA(columnsA), rowsB(rowsB), columnsB(columnsB) {
      assert(rowsA != 0 && columnsA != 0 && rowsB && columnsB != 0 &&
             "expecting valid shape");
    }

    unsigned rowsA = 0;
    unsigned columnsA = 0;
    unsigned rowsB = 0;
    unsigned columnsB = 0;
  };

  TargetArchNativeSizes() = default;

  void setDotShape(unsigned bitWidth, DotShape shape) {
    dotShapes[bitWidth] = shape;
  }
  void setBlockMemShape(unsigned bitWidth, BlockMemShape shape) {
    blockMemShapes[bitWidth] = shape;
  }
  void setLoadStoreSize(unsigned size) { loadStoreSize = size; }
  const DotShape &getDotShape(unsigned bitWidth) const {
    assert(dotShapes.contains(bitWidth) &&
           "No dot shape configured for bit width");
    return dotShapes.at(bitWidth);
  }
  const BlockMemShape &getBlockMemShape(unsigned bitWidth) const {
    assert(blockMemShapes.contains(bitWidth) &&
           "No block memory access shape configured for bit width");
    return blockMemShapes.at(bitWidth);
  }
  unsigned getLoadStoreSize() const { return loadStoreSize; }

private:
  /// Stores the natively supported dot shape per bitwidth of the operand data
  /// type, e.g. 16 -> 8x16x16 (MxKxN) for [b]float16 on PVC.
  llvm::SmallDenseMap<unsigned, DotShape> dotShapes;
  /// Stores the natively supported shapes for 2D block reads of dot operands,
  /// per element type bitwidth.
  llvm::SmallDenseMap<unsigned, BlockMemShape> blockMemShapes;
  unsigned loadStoreSize = 0;
};

class MatchTargetSizePass
    : public triton::gpu::intel::impl::TritonIntelGPUMatchTargetSizeBase<
          MatchTargetSizePass> {
public:
  void runOnOperation() override {
    initNativeOperationSizes();

    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();

    // Collect the result layout of "interesting" `tt.dot` operations.
    // A candidate 'tt.dot' operation yields a tensor with a warp layout.
    m.walk([&](tt::DotOp dot) {
      auto resultType = cast<RankedTensorType>(dot.getResult().getType());
      if (isCandidate(resultType))
        dotAttrs.insert(resultType.getEncoding());
    });

    // Split operations to match the target architecture native shapes.
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      SmallVector<Type> types(op->getOperandTypes().begin(),
                              op->getOperandTypes().end());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());
      types.append(resultTypes);

      if (llvm::none_of(types, [this](Type type) { return isCandidate(type); }))
        return WalkResult::advance();
      if (isa<scf::ForOp, scf::YieldOp>(op))
        return WalkResult::advance();

      LLVM_DEBUG({
        llvm::dbgs() << "Processing operation: " << *op << "\n";
        llvm::dbgs() << "Module before transformation:\n" << m << "\n\n";
      });

      if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        recordRootSubSize(cstOp.getResult().getType());
        transformArithConstantOp(cstOp);
      } else if (auto ptrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        recordRootSubSize(ptrOp.getResult().getType());
        transformMakeTensorPtrOp(ptrOp);
      } else if (auto dot = dyn_cast<tt::DotOp>(op))
        transformDotOp(dot);
      else
        transformGenericOp(op);

      LLVM_DEBUG({
        llvm::dbgs() << "Module after transformation:\n" << m << "\n\n";
      });

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "Canonicalizing...\n");
    canonicalize();
    LLVM_DEBUG(llvm::dbgs() << "Module after canonicalization:\n"
                            << m << "\n\n");

    // By default, tritongpu are lowered to simt mode (threads-per-warp=16)
    // instead of simd mode (threads-per-warp=1).
    // FIXME: force threads-per-warp=16 in simt(this should be done via an
    // analysis designed to determine whether the kernel contains tt.dot
    // operations that use block pointers).
    m->setAttr("triton_gpu.threads-per-warp",
               IntegerAttr::get(IntegerType::get(ctx, 32), 16));
  }

private:
  /// Initialize the native operation sizes supported by the target
  /// architecture.
  void initNativeOperationSizes();

  /// Determine whether the given type is a tensor (or a pointer to a tensor)
  /// that has a warp layout or a dot layout with a parent warp layout.
  bool isCandidate(Type type) const;

  /// Canonicalize operations (e.g. remove redundant tt.extract, tt.glue)
  void canonicalize();

  void recordRootSubSize(Type type);
  SmallVector<int64_t> getSubOpSize(RankedTensorType type) const;
  std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
  getSubTypeAndShape(Type type) const;

  /// Transformations for specific operations.
  void transformMakeTensorPtrOp(tt::MakeTensorPtrOp op);
  void transformArithConstantOp(arith::ConstantOp op);
  void transformDotOp(tt::DotOp dot);

  /// Generic transformation.
  void transformGenericOp(Operation *op);

  /// Record the native size supported by the target implementation.
  DenseMap<Attribute, SmallVector<int64_t>> sizePerAttrMap;

  /// Collects the result layout of the `tt.dot` operations in the module.
  DenseSet<Attribute> dotAttrs;

  /// The native operation sizes supported by the target architecture.
  TargetArchNativeSizes nativeSizes;
};

/// Simplify arith operations with constant RHS.
class ArithRemPattern : public OpRewritePattern<arith::RemSIOp> {
public:
  using OpRewritePattern<arith::RemSIOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemSIOp op,
                                PatternRewriter &rewriter) const final {
    APInt value;
    if (!matchPattern(op.getRhs(), m_ConstantInt(&value)))
      return failure();

    Location loc = op.getLoc();
    Type type = op.getType();

    // lhs % 0x00100000 -> lhs & (0x00100000-1)
    if (value.popcount() == 1) {
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

/// Simplify SCF for loops.
/// Before:
///   %glue = triton_intel_gpu.glue %a, %b : tensor<4x4xf32>, tensor<4x4xf32>
///         -> tensor<8x4xf32>
///   scf.for %i = %lb to %ub step %step (%arg = %glue) {
///     %extract = triton_intel_gpu.extract %arg[0] : tensor<8x4xf32>
///              -> tensor<4x4xf32>
///     use %extract
///     scf.yield
///   }
/// After:
///   scf.for %i = %lb to %ub step %step (%arg = %a) {
///     use %arg
///     scf.yield
///   }
class ScfPattern : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    if (!isCandidateLoop(forOp))
      return failure();

    SmallVector<Operation *> deleteList;
    SmallVector<Value> newInits;
    DenseMap<Value, int> userIndexMap;
    unsigned idx = 0;

    // Create a new initialization list by replacing 'glue' operations with
    // their operands. Record 'extract' operations that use original init
    // argument so that they can be updated after the loop init list is
    // expanded.
    for (auto [arg, init] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
      Operation *definingOp = init.getDefiningOp();
      if (!isa_and_nonnull<ttgi::GlueOp>(definingOp)) {
        newInits.push_back(init);
        userIndexMap[arg] = idx++;
        continue;
      }

      auto glue = cast<ttgi::GlueOp>(definingOp);
      unsigned numSplit = glue->getOperands().size();
      for (unsigned i = 0; i < numSplit; ++i)
        newInits.push_back(glue->getOperand(i));

      for (Operation *user : arg.getUsers()) {
        if (auto extract = dyn_cast<ttgi::ExtractOp>(user)) {
          userIndexMap[extract] = idx + extract.getIndex();
          deleteList.push_back(extract.getOperation());
        }
      }
      idx += numSplit;
    }

    if (newInits.size() == forOp.getInits().size())
      return failure();

    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInits);

    for (auto [user, idx] : userIndexMap)
      user.replaceAllUsesWith(newForOp.getRegionIterArgs()[idx]);

    forOp.getInductionVar().replaceAllUsesWith(newForOp.getInductionVar());

    // Copy the loop body over to the new loop.
    Block *body = newForOp.getBody();
    body->getOperations().splice(body->begin(),
                                 forOp.getBody()->getOperations());

    // Replace the yield op in the new loop.
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    SmallVector<Value> newValues;
    for (auto result : yield.getResults())
      if (Operation *def = result.getDefiningOp()) {
        if (auto glue = dyn_cast<ttgi::GlueOp>(def))
          newValues.append(glue->getOperands().begin(),
                           glue->getOperands().end());
        else
          newValues.push_back(result);
      }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(yield);
    rewriter.create<scf::YieldOp>(yield.getLoc(), newValues);
    rewriter.eraseOp(yield);

    // Replace uses of the original loop results with the new loop results.
    userIndexMap.clear();
    idx = 0;
    for (auto [result, init] :
         llvm::zip(forOp.getResults(), forOp.getInits())) {
      Operation *definingOp = init.getDefiningOp();
      if (!isa_and_nonnull<ttgi::GlueOp>(definingOp)) {
        userIndexMap[result] = idx++;
        continue;
      }

      auto glue = cast<ttgi::GlueOp>(definingOp);
      for (Operation *user : result.getUsers())
        if (auto extract = dyn_cast<ttgi::ExtractOp>(user)) {
          userIndexMap[extract] = idx + extract.getIndex();
          deleteList.push_back(extract.getOperation());
        }

      idx += glue->getOperands().size();
    }

    for (auto [user, idx] : userIndexMap)
      user.replaceAllUsesWith(newForOp.getResults()[idx]);

    for (Operation *deleteOp : deleteList)
      rewriter.eraseOp(deleteOp);

    rewriter.eraseOp(forOp);

    return success();
  }

  bool isCandidateLoop(scf::ForOp forOp) const {
    // If none of the loop init values is defined by a 'glue' operation there is
    // nothing to do.
    if (llvm::none_of(forOp.getInits(), [](Value init) {
          return init.getDefiningOp() &&
                 isa<ttgi::GlueOp>(init.getDefiningOp());
        })) {
      return false;
    }

    // Bail out if any user of a 'glue' init value is not an 'extract'
    // operation.
    for (auto [arg, init] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
      if (!init.getDefiningOp() || !isa<ttgi::GlueOp>(init.getDefiningOp()))
        continue;

      if (llvm::any_of(arg.getUsers(), [](Operation *user) {
            return !isa<ttgi::ExtractOp>(user);
          })) {
        return false;
      }
    }

    // Bail out if any user of a loop result is not an 'extract' operation
    // (otherwise we would have to materialize a 'glue' operation after the loop
    // is replaced, which complicates things).
    for (OpResult result : forOp->getResults()) {
      if (llvm::any_of(result.getUsers(), [](Operation *user) {
            return !isa<ttgi::ExtractOp>(user);
          })) {
        return false;
      }
    }

    return true;
  }
};

void MatchTargetSizePass::initNativeOperationSizes() {
  // FIXME: sets the target dot shape natively supported by the target
  // architecture using the target architecture information when available.
  // These values works for PVC.

  // clang-format off
  //                bitwidth   M   N   K
  nativeSizes.setDotShape( 8, {8, 16, 32});
  nativeSizes.setDotShape(16, {8, 16, 16});
  nativeSizes.setDotShape(32, {8, 16,  8});

  //                     bitwidth   rA  cA  rB  cB
  nativeSizes.setBlockMemShape( 8, {16, 64, 32, 32});
  nativeSizes.setBlockMemShape(16, {32, 32, 32, 32});
  nativeSizes.setBlockMemShape(32, { 8,  8,  8, 16});

  nativeSizes.setLoadStoreSize(512); // max 512DW;
  // clang-format on
}

bool MatchTargetSizePass::isCandidate(Type type) const {
  auto isTensorWithWarpLayout = [](Type type) {
    auto isCandidateLayout = [](Attribute layout) {
      if (!layout)
        return false;
      if (isa<ttgi::WarpEncodingAttr>(layout))
        return true;
      if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout))
        return isa<ttgi::WarpEncodingAttr>(dotLayout.getParent());
      return false;
    };

    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return isCandidateLayout(tensorType.getEncoding());
    return false;
  };

  if (isTensorWithWarpLayout(type))
    return true;

  if (auto ptrType = dyn_cast<tt::PointerType>(type))
    return isTensorWithWarpLayout(ptrType.getPointeeType());

  return false;
}

void MatchTargetSizePass::canonicalize() {
  MLIRContext *ctx = &getContext();
  ModuleOp m = getOperation();

  RewritePatternSet patterns(ctx);
  patterns.add<ScfPattern>(ctx);
  patterns.add<ArithRemPattern>(ctx); // FIXME: upstream to arith dialect.

  if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns))))
    signalPassFailure();
}

void MatchTargetSizePass::recordRootSubSize(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Attribute layout = tensorType.getEncoding();
    assert(layout && "Expecting a valid layout");
    if (sizePerAttrMap.count(layout) == 0)
      sizePerAttrMap[layout] = getSubOpSize(tensorType);
    return;
  }

  if (auto ptrType = dyn_cast<tt::PointerType>(type))
    recordRootSubSize(ptrType.getPointeeType());
}

/// Return the native size supported by the target architecture.
SmallVector<int64_t>
MatchTargetSizePass::getSubOpSize(RankedTensorType type) const {
  Attribute layout = type.getEncoding();
  assert(layout && "Expecting a valid layout");

  // Dot operation.
  if (dotAttrs.count(layout)) {
    const TargetArchNativeSizes::DotShape &dotShape =
        nativeSizes.getDotShape(type.getElementTypeBitWidth());
    SmallVector<int64_t> nativeDotSize{dotShape.m, dotShape.n};
    return nativeDotSize;
  }

  // Load/Store operations.
  ArrayRef<int64_t> shape = type.getShape();
  const unsigned sizeInBits = type.getElementTypeBitWidth();
  const unsigned sizeInBytes = sizeInBits / 8;
  unsigned maxLoadStoreSize = nativeSizes.getLoadStoreSize();

  SmallVector<int64_t> subSize(shape.size());
  switch (shape.size()) {
  case 1: {
    int64_t max = maxLoadStoreSize * 4 / sizeInBytes;
    subSize[0] = std::min(max, shape[0]);
  } break;
  case 2: {
    if (isa<ttgi::WarpEncodingAttr>(layout)) {
      // 32 = 2 * 16(subgroupSize) which is for large load/store
      subSize[1] = std::min(32L, shape[1]);
      // FIXME: From gfxspec, max 2d block load height is 32
      subSize[0] = std::min(32L, shape[0]);
    } else if (auto dotLayout = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
      const TargetArchNativeSizes::BlockMemShape &memShape =
          nativeSizes.getBlockMemShape(sizeInBits);
      switch (dotLayout.getOpIdx()) {
      case 0:
        subSize[1] =
            std::min(static_cast<int64_t>(memShape.columnsA), shape[1]);
        subSize[0] = std::min(static_cast<int64_t>(memShape.rowsA), shape[0]);
        break;
      case 1:
        subSize[1] =
            std::min(static_cast<int64_t>(memShape.columnsB), shape[1]);
        subSize[0] = std::min(static_cast<int64_t>(memShape.rowsB), shape[0]);
        break;
      }
    } else {
      llvm_unreachable("Unsupported layout");
    }
  } break;
  default:
    llvm_unreachable("Unsupported shape");
  }

  return subSize;
}

/// return [shape, subType, subSize] for a tensor (or pointer to tensor)
std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
MatchTargetSizePass::getSubTypeAndShape(Type type) const {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Attribute layout = tensorType.getEncoding();
    assert(layout && "Expecting a valid layout");
    SmallVector<int64_t> shape = to_vector(tensorType.getShape());
    SmallVector<int64_t> subSize = sizePerAttrMap.at(layout);
    auto subType = RankedTensorType::get(
        subSize, tensorType.getElementType() /*no encoding*/);
    return {shape, subType, subSize};
  }

  if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    auto [shape, subType, subSize] = getSubTypeAndShape(pointeeType);
    auto newType = tt::PointerType::get(subType, ptrType.getAddressSpace());
    return {shape, newType, subSize};
  }

  return {{0}, type, {0}};
}

void MatchTargetSizePass::transformMakeTensorPtrOp(tt::MakeTensorPtrOp op) {
  Type resultType = op.getResult().getType();
  auto [shape, subType, subSize] = getSubTypeAndShape(resultType);
  unsigned dim = shape.size();
  OpBuilder b(op);
  Location loc = op.getLoc();

  SmallVector<Value> subOps;
  for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
    switch (dim) {
    case 2: {
      for (unsigned j = 0; j < shape[0]; j += subSize[0]) {
        auto offsets = op.getOffsets();
        // newOffsets = offsets += [j, i]
        SmallVector<Value> newOffsets(2);
        newOffsets[0] = (j == 0)
                            ? offsets[0]
                            : b.create<arith::AddIOp>(
                                  loc, offsets[0],
                                  b.create<arith::ConstantIntOp>(loc, j, 32));
        newOffsets[1] = (i == 0)
                            ? offsets[1]
                            : b.create<arith::AddIOp>(
                                  loc, offsets[1],
                                  b.create<arith::ConstantIntOp>(loc, i, 32));

        SmallVector<int32_t> subShape;
        for (int64_t sub : subSize)
          subShape.push_back(sub);

        auto subOp = b.create<tt::MakeTensorPtrOp>(
            loc, op.getBase(), op.getShape(), op.getStrides(), newOffsets,
            subShape, op.getOrder());
        subOps.push_back(subOp);
      }
    } break;
    default:
      llvm_unreachable("Unsupported shape");
    }
  }

  op->replaceAllUsesWith(
      b.create<ttgi::GlueOp>(loc, resultType, subOps)->getResults());
  op->erase();
}

void MatchTargetSizePass::transformArithConstantOp(arith::ConstantOp op) {
  Type resultType = cast<RankedTensorType>(op.getResult().getType());
  auto [shape, subType, subSize] = getSubTypeAndShape(resultType);
  unsigned dim = shape.size();
  OpBuilder b(op);
  Location loc = op.getLoc();

  auto value = cast<DenseElementsAttr>(op.getValue());
  value = value.resizeSplat(cast<ShapedType>(subType));
  SmallVector<Value> subOps;

  for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
    switch (dim) {
    case 2: {
      for (unsigned j = 0; j < shape[0]; j += subSize[0]) {
        auto subOp = b.create<arith::ConstantOp>(loc, subType, value);
        subOps.push_back(subOp);
      }
    } break;
    default:
      llvm_unreachable("Unsupported type shape");
    }
  }

  op->replaceAllUsesWith(
      b.create<ttgi::GlueOp>(loc, resultType, subOps)->getResults());
  op->erase();
}

void MatchTargetSizePass::transformDotOp(tt::DotOp dot) {
  auto aType = cast<RankedTensorType>(dot.getA().getType());
  auto bType = cast<RankedTensorType>(dot.getB().getType());
  auto cType = cast<RankedTensorType>(dot.getC().getType());
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  int64_t m = aShape[0];
  int64_t n = bShape[1];
  int64_t k = aShape[1];
  const TargetArchNativeSizes::DotShape &dotShape =
      nativeSizes.getDotShape(aType.getElementTypeBitWidth());

  OpBuilder b(dot);
  Location loc = dot.getLoc();

  auto getSubDotVal = [&](Value val, int64_t mm, int64_t kk, int64_t mStep,
                          int64_t kStep) {
    auto [shape, subType, subSize] = getSubTypeAndShape(val.getType());
    unsigned subIdx =
        (kk / subSize[1]) * (shape[0] / subSize[0]) + mm / subSize[0];
    Value subVal = b.create<ttgi::ExtractOp>(loc, subType, val, subIdx);
    auto subDotType = RankedTensorType::get(
        {mStep, kStep}, cast<RankedTensorType>(val.getType()).getElementType());
    unsigned subDotIdx = ((kk % subSize[1]) / kStep) * (subSize[0] / mStep) +
                         (mm % subSize[0]) / mStep;
    return b.create<ttgi::ExtractOp>(loc, subDotType, subVal, subDotIdx);
  };

  auto [shape, subType, subSize] = getSubTypeAndShape(cType);
  SmallVector<Value> subCs;
  for (unsigned nn = 0; nn < n; nn += dotShape.n) {
    for (unsigned mm = 0; mm < m; mm += dotShape.m) {
      Value subDotC = getSubDotVal(dot.getC(), mm, nn, dotShape.m, dotShape.n);
      for (unsigned kk = 0; kk < k; kk += dotShape.k) {
        Value subDotA =
            getSubDotVal(dot.getA(), mm, kk, dotShape.m, dotShape.k);
        Value subDotB =
            getSubDotVal(dot.getB(), kk, nn, dotShape.k, dotShape.n);
        subDotC = b.create<tt::DotOp>(loc, subDotA, subDotB, subDotC,
                                      dot.getInputPrecisionAttr(),
                                      dot.getMaxNumImpreciseAccAttr());
      }
      subCs.push_back(subDotC);
    }
  }

  dot->replaceAllUsesWith(
      b.create<ttgi::GlueOp>(loc, dot.getType(), subCs)->getResults());
  dot->erase();
}

void MatchTargetSizePass::transformGenericOp(Operation *op) {
  unsigned numResults = op->getResults().size();
  unsigned dotIdx = 2;
  Type type;

  switch (numResults) {
  case 0:
    // prefetch/store
    type = op->getOperand(0).getType();
    break;
  case 1: {
    // arith/math/advanceOp/loadOp
    type = op->getResultTypes()[0];
    // mark tt.load for dot A/B
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      if (isa<tt::LoadOp>(op)) {
        Attribute layout = tensorType.getEncoding();
        assert(layout && "Expecting a valid layout");
        if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout))
          dotIdx = dotAttr.getOpIdx();
      }
  } break;
  default:
    llvm_unreachable("Unexpected operation");
  }

  auto [shape, subType, subSize] = getSubTypeAndShape(type);
  unsigned dim = shape.size();
  OpBuilder b(op);
  Location loc = op->getLoc();
  unsigned idx = 0;

  SmallVector<Value> subOps;
  for (unsigned i = 0; i < shape[dim - 1]; i += subSize[dim - 1]) {
    switch (dim) {
    case 2: {
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
                          }
                          return operand;
                        });
        Operation *subOp;
        if (numResults == 0)
          subOp = b.create(loc, op->getName().getIdentifier(), newOperands, {},
                           op->getAttrs());
        else {
          subOp = b.create(loc, op->getName().getIdentifier(), newOperands,
                           subType, op->getAttrs());
          if (dotIdx < 2)
            subOp->setAttr("DotIdx", b.getIntegerAttr(b.getI32Type(), dotIdx));
          subOps.push_back(subOp->getResults()[0]);
        }
        ++idx;
      }
    } break;
    default:
      llvm_unreachable("Unsupported shape");
    }
  }

  if (numResults == 1) {
    op->replaceAllUsesWith(b.create<ttgi::GlueOp>(loc, type, subOps));
  }
  op->erase();
}

} // namespace
