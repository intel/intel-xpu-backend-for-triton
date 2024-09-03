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
#include "triton/Tools/Sys/GetEnv.hpp"

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

  TargetArchNativeSizes() = default;
  TargetArchNativeSizes(DotShape dotShape, unsigned loadStoreSize)
      : dotShape(dotShape), loadStoreSize(loadStoreSize) {}

  void setDotShape(DotShape shape) { dotShape = shape; }
  void setLoadStoreSize(unsigned size) { loadStoreSize = size; }
  const DotShape &getDotShape() const { return dotShape; }
  unsigned getLoadStoreSize() const { return loadStoreSize; }

private:
  DotShape dotShape;
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

    // this is ad-hoc for Q SLM
    if (mlir::triton::tools::getBoolEnv("ENABLE_SLM")) {
      tmpSet.clear();
      tt::LoadOp load;
      m.walk([&](tt::LoadOp op) {
        load = op;
        return WalkResult::interrupt();
      });
      auto dot = cast<tt::DotOp>(*load->getUsers().begin());
      auto type = cast<RankedTensorType>(load.getType());
      unsigned bytes =
          type.getNumElements() * type.getElementTypeBitWidth() / 8;
      unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(m);
      unsigned slmSize = numWarps * bytes;
      m->setAttr(
          "triton_gpu.shared",
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), slmSize));
      auto func = load->getParentOfType<FunctionOpInterface>();
      auto slmTy = tt::PointerType::get(type.getElementType(), 3);
      func.insertArgument(func.getNumArguments(), slmTy, {}, func.getLoc());
      Location loc = load.getLoc();
      OpBuilder b(load);
      b.setInsertionPointAfter(load);
      auto subgroupId = b.create<gpu::SubgroupIdOp>(loc);
      auto warpId =
          b.create<arith::IndexCastOp>(loc, b.getI32Type(), subgroupId);
      // hardcode: we use i16ptr here, so bytes / 2
      auto warpSize = b.create<arith::ConstantIntOp>(loc, bytes / 2, 32);
      auto offset = b.create<arith::MulIOp>(loc, warpId, warpSize);
      // auto arg = func.getArgument(func.getNumArguments() - 1);
      auto block = load->getBlock();
      auto arg = block->getArgument(block->getNumArguments() - 1);
      auto base = b.create<tt::AddPtrOp>(loc, slmTy, arg, offset);
      SmallVector<Value> shape;
      shape.push_back(
          b.create<arith::ConstantIntOp>(loc, type.getShape()[0], 64));
      shape.push_back(
          b.create<arith::ConstantIntOp>(loc, type.getShape()[1], 64));
      SmallVector<Value> strides;
      strides.push_back(
          b.create<arith::ConstantIntOp>(loc, type.getShape()[1], 64));
      strides.push_back(b.create<arith::ConstantIntOp>(loc, 1, 64));
      SmallVector<Value> offsets;
      offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
      offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
      auto loadPtrTy = cast<tt::PointerType>(load.getPtr().getType());
      auto ptrTy = tt::PointerType::get(loadPtrTy.getPointeeType(), 3);
      auto ptr = b.create<tt::MakeTensorPtrOp>(loc, ptrTy, base, shape, strides,
                                               offsets,
                                               b.getDenseI32ArrayAttr({1, 0}));
      auto store = b.create<tt::StoreOp>(
          loc, ptr, load, tt::CacheModifier::NONE, tt::EvictionPolicy::NORMAL);

      //
      b.setInsertionPoint(dot);
      auto newLoad =
          b.create<tt::LoadOp>(dot.getLoc(), ptr, tt::CacheModifier::NONE,
                               tt::EvictionPolicy::NORMAL, false);
      Value res = load.getResult();
      tmpSet.insert(dot);
      res.replaceAllUsesExcept(newLoad, store);
    }

    // Collect the result layout of "interesting" `tt.dot` operations.
    // A candidate 'tt.dot' operation yields a tensor with a warp layout.
    m.walk([&](tt::DotOp dot) {
      auto resultType = cast<RankedTensorType>(dot.getResult().getType());
      if (isCandidate(resultType))
        dotAttrs.insert(resultType.getEncoding());
    });

    auto hasSliceAttr = [](Type type) {
      auto tType = dyn_cast<RankedTensorType>(type);
      if (tType && isa<ttg::SliceEncodingAttr>(tType.getEncoding()))
        return true;
      return false;
    };

    // Split operations to match the target architecture native shapes.
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      SmallVector<Type> types(op->getOperandTypes().begin(),
                              op->getOperandTypes().end());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());
      types.append(resultTypes);

      if (llvm::none_of(types, [this](Type type) { return isCandidate(type); }))
        ;
      else if (isa<scf::ForOp, scf::YieldOp>(op))
        ;
      // FIXME: hack it for now
      else if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op))
        convert.getResult().replaceAllUsesWith(convert.getSrc());
      else if (auto reduce = dyn_cast<tt::ReduceOp>(op))
        transformReduceOp(reduce);
      else if (op->getNumResults() == 1 &&
               hasSliceAttr(op->getResultTypes()[0]))
        ;
      else if (auto expand = dyn_cast<tt::ExpandDimsOp>(op))
        ;
      else if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        recordRootSubSize(cstOp.getResult().getType());
        transformArithConstantOp(cstOp);
      } else if (auto ptrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        recordRootSubSize(ptrOp.getResult().getType());
        transformMakeTensorPtrOp(ptrOp);
      } else if (auto dot = dyn_cast<tt::DotOp>(op))
        transformDotOp(dot);
      else if (auto bc = dyn_cast<tt::BroadcastOp>(op))
        transformBroadcastOp(bc);
      // arith,math,tt.advance,tt.load,tt.store,tt.prefetch
      // tt.splat, tt.broadcast
      else
        transformGenericOp(op);
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

    llvm::errs() << m;
  }

private:
  DenseSet<Value> tmpSet;
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
  getSubTypeAndShape(Type type, bool hack = false) const;
  Value getSubVal(Operation *op, Value val, ArrayRef<int64_t> srcOffset,
                  ArrayRef<int64_t> dstSize);

  /// Transformations for specific operations.
  void transformMakeTensorPtrOp(tt::MakeTensorPtrOp op);
  void transformArithConstantOp(arith::ConstantOp op);
  void transformDotOp(tt::DotOp dot);
  void transformReduceOp(tt::ReduceOp op);
  void transformBroadcastOp(tt::BroadcastOp op);

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

class ScfPattern : public OpRewritePattern<scf::ForOp> {
public:
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const final {
    SmallVector<Operation *> deleteList;
    SmallVector<Value> newInits;
    DenseMap<Value, int> userIndexMap;
    auto idx = 0;
    for (auto [arg, init] : llvm::zip(op.getRegionIterArgs(), op.getInits())) {
      auto glue = dyn_cast<ttgi::GlueOp>(init.getDefiningOp());
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
        auto extract = dyn_cast<ttgi::ExtractOp>(user);
        if (extract) {
          userIndexMap[extract] = idx + extract.getIndex();
          deleteList.push_back(extract.getOperation());
        }
      }
      idx += numSplit;
    }
    if (newInits.size() == op.getInits().size())
      return failure();
    auto newOp =
        rewriter.create<scf::ForOp>(op.getLoc(), op.getLowerBound(),
                                    op.getUpperBound(), op.getStep(), newInits);

    for (auto [user, idx] : userIndexMap)
      user.replaceAllUsesWith(newOp.getRegionIterArgs()[idx]);
    op.getInductionVar().replaceAllUsesWith(newOp.getInductionVar());
    // splice operations
    auto *body = newOp.getBody();
    body->getOperations().splice(body->begin(), op.getBody()->getOperations());
    // yield op
    auto yield = cast<scf::YieldOp>(body->getTerminator());
    SmallVector<Value> newValues;
    for (auto result : yield.getResults()) {
      auto def = result.getDefiningOp();
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
          userIndexMap[extract] = idx + extract.getIndex();
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
/*
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
*/

void MatchTargetSizePass::initNativeOperationSizes() {
  // FIXME: sets the target dot shape natively supported by the target
  // architecture using the target architecture information when available.
  // These value works for PVC.
  TargetArchNativeSizes::DotShape shape(8, 16, 16);
  nativeSizes.setDotShape(shape);
  nativeSizes.setLoadStoreSize(512); // max 512DW;
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

  int64_t colLimit = 0;
  const auto &dotShape = nativeSizes.getDotShape();
  // Dot operation.
  if (dotAttrs.count(layout)) {
    SmallVector<int64_t> nativeDotSize{dotShape.m, dotShape.n};
    return nativeDotSize;
    // 32 = 2 * 16(subgroupSize) which is for large load/store
  } else if (auto warpAttr = dyn_cast<ttgi::WarpEncodingAttr>(layout)) {
    colLimit = 32;
  } else if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    if (dotAttr.getKWidth() != 0 && dotAttr.getOpIdx() == 1)
      return {dotShape.k, dotShape.n};
    colLimit = 32;
    // hack for attn
    if (dotAttr.getOpIdx() == 1)
      colLimit = 16;
  }

  // Load/Store operations.
  ArrayRef<int64_t> shape = type.getShape();
  const unsigned sizeInBytes = type.getElementTypeBitWidth() / 8;
  unsigned maxLoadStoreSize = nativeSizes.getLoadStoreSize();

  SmallVector<int64_t> subSize(shape.size());
  switch (shape.size()) {
  case 1: {
    int64_t max = maxLoadStoreSize * 4 / sizeInBytes;
    subSize[0] = std::min(32L, shape[0]);
  } break;
  case 2: {
    subSize[1] = (shape[1] > colLimit) ? colLimit : shape[1];
    int64_t max = maxLoadStoreSize * 4 / sizeInBytes / subSize[1];
    subSize[0] = std::min(32L, shape[0]);
  } break;
  default:
    llvm_unreachable("Unsupported shape");
  }

  return subSize;
}

/// FIXME: add a map for look up
/// return [shape, subType, subSize] for a tensor (or pointer to tensor)
std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
MatchTargetSizePass::getSubTypeAndShape(Type type, bool hack) const {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Attribute layout = tensorType.getEncoding();
    SmallVector<int64_t> shape = to_vector(tensorType.getShape());
    SmallVector<int64_t> subSize =
        layout ? sizePerAttrMap.at(layout) : SmallVector<int64_t>{shape[0], 32};
    // hack for attn
    if (hack) {
      subSize[0] = 8;
      subSize[1] = 16;
    }

    auto subType = RankedTensorType::get(
        subSize, tensorType.getElementType() /*no encoding*/);
    return {shape, subType, subSize};
  }

  if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    auto [shape, subType, subSize] = getSubTypeAndShape(pointeeType, hack);
    auto newType = tt::PointerType::get(subType, ptrType.getAddressSpace());
    return {shape, newType, subSize};
  }

  return {{0}, type, {0}};
}

Value MatchTargetSizePass::getSubVal(Operation *op, Value val,
                                     ArrayRef<int64_t> srcOffset,
                                     ArrayRef<int64_t> dstSize) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  auto elmTy = cast<RankedTensorType>(val.getType()).getElementType();
  auto [shape, subType, subSize] = getSubTypeAndShape(val.getType());
  unsigned srcIdx = (srcOffset[1] / subSize[1]) * (shape[0] / subSize[0]) +
                    srcOffset[0] / subSize[0];
  Value subSrcVal = b.create<ttgi::ExtractOp>(loc, subType, val, srcIdx);
  assert(dstSize[0] <= subSize[0] && "add more support");
  unsigned dstIdx =
      ((srcOffset[1] % subSize[1]) / dstSize[1]) * (subSize[0] / dstSize[0]) +
      (srcOffset[0] % subSize[0]) / dstSize[0];
  auto dstType = dstSize[0] == 1 ? RankedTensorType::get(dstSize[1], elmTy)
                                 : RankedTensorType::get(dstSize, elmTy);
  Value dstVal = b.create<ttgi::ExtractOp>(loc, dstType, subSrcVal, dstIdx);
  return dstVal;
}

static Value hackAlloc(OpBuilder &b, Location loc, Type ptrTy, int64_t size) {
  auto func = static_cast<FunctionOpInterface>(
      &*b.getInsertionPoint()
            ->getParentWithTrait<FunctionOpInterface::Trait>());
  auto m = func->getParentOfType<ModuleOp>();
  constexpr StringLiteral SharedAttrName = "triton_gpu.shared";
  if (!m->getAttr(SharedAttrName)) {
    m->setAttr(SharedAttrName, b.getIndexAttr(size));
    func.insertArgument(func.getNumArguments(), ptrTy, b.getDictionaryAttr({}),
                        loc);
  }
  return func.getArguments().back();
}

void MatchTargetSizePass::transformReduceOp(tt::ReduceOp op) {
  auto loc = op.getLoc();
  OpBuilder b(op);
  assert(op.getSrcs().size() == 1 && "only support one src");
  auto src = op.getSrcs().front();
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto dims = srcTy.getShape().size();
  auto axis = op.getAxis();
  assert(axis == dims - 1 && "only support last axis");
  assert(dims <= 2 && "only support 1D/2D tensor");
  auto outer = dims == 2 ? srcTy.getShape()[0] : 1;
  auto combine = op.getCombineOp().front().getOperations().begin();
  auto id = combine->getName().getIdentifier();

  SmallVector<Value> glueVals;
  unsigned step = 16;

  for (unsigned i = 0; i < outer; i += step) {
    SmallVector<Value> subVals;
    RankedTensorType dstType = RankedTensorType::get(
        {srcTy.getShape()[0], step}, srcTy.getElementType());
    RankedTensorType subGlueType = RankedTensorType::get(
        {srcTy.getShape()[0] / 2, step}, srcTy.getElementType());
    for (unsigned j = 0; j < srcTy.getShape()[axis]; j += step) {
      std::array<Value, 2> subGlues{
          b.create<ttgi::ExtractOp>(loc, subGlueType, src, j / step * 2),
          b.create<ttgi::ExtractOp>(loc, subGlueType, src, j / step * 2 + 1)};
      Value subVal = b.create<ttgi::GlueOp>(loc, dstType, subGlues);
      subVals.push_back(subVal);
    }
    auto subType = dstType;

    Value acc;
    switch (subVals.size()) {
    case 1:
      acc = subVals[0];
      break;
    case 2: {
      auto acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      acc = acc01->getResult(0);
      break;
    }
    case 4: {
      auto acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      auto acc23 = b.create(loc, id, {subVals[2], subVals[3]}, subType);
      auto accOp = b.create(loc, id, {acc01->getResult(0), acc23->getResult(0)},
                            subType);
      acc = accOp->getResult(0);
      break;
    }
    default:
      assert(false && "add more reduce size support");
    }
    auto m = op->getParentOfType<ModuleOp>();
    // Fixed size for num_warps matrices of sg_size^2 shape.
    int64_t size = step * step * srcTy.getElementTypeBitWidth() / 8 *
                   ttg::TritonGPUDialect::getNumWarps(m);
    Type allocTy = cast<RankedTensorType>(acc.getType()).getElementType();
    Type ptrTy = tt::PointerType::get(allocTy, tt::TritonGEN::kWorkgroup);
    Value localBuffer = hackAlloc(b, loc, ptrTy, size);
    Value accT =
        b.create<ttgi::SubGroupTranspose>(loc, acc.getType(), localBuffer, acc);
    auto sgReduce = b.create<tt::ReduceOp>(loc, accT, axis);
    Region &sgReduceRegion = sgReduce.getCombineOp();
    b.cloneRegionBefore(op.getCombineOp(), sgReduceRegion,
                        sgReduceRegion.end());
    glueVals.push_back(sgReduce->getResult(0));
  }

  auto resultType = cast<RankedTensorType>(op.getResultTypes()[0]);
  RankedTensorType glueType =
      RankedTensorType::get(resultType.getShape(), resultType.getElementType());

  Value res = glueVals.size() == 1
                  ? glueVals.front()
                  : b.create<ttgi::GlueOp>(loc, glueType, glueVals).getRes();
  res = b.create<ttg::ConvertLayoutOp>(loc, resultType, res);

  op->replaceAllUsesWith(ValueRange{res});
  op->erase();
}

void MatchTargetSizePass::transformMakeTensorPtrOp(tt::MakeTensorPtrOp op) {
  Type resultType = op.getResult().getType();
  bool hack = false;
  if (cast<tt::PointerType>(resultType).getAddressSpace() == 3) {
    hack = true;
  }
  auto [shape, subType, subSize] = getSubTypeAndShape(resultType, hack);
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

        Value subOp = b.create<tt::MakeTensorPtrOp>(
            loc, op.getBase(), op.getShape(), op.getStrides(), newOffsets,
            subShape, op.getOrder());
        subOp.setType(subType);
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
  value = value.resizeSplat(subType.cast<ShapedType>());
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
  auto aType = dot.getA().getType().cast<RankedTensorType>();
  auto bType = dot.getB().getType().cast<RankedTensorType>();
  auto cType = dot.getC().getType().cast<RankedTensorType>();
  ArrayRef<int64_t> aShape = aType.getShape();
  ArrayRef<int64_t> bShape = bType.getShape();
  int64_t m = aShape[0];
  int64_t n = bShape[1];
  int64_t k = aShape[1];
  const auto &dotShape = nativeSizes.getDotShape();
  OpBuilder b(dot);
  Location loc = dot.getLoc();

  auto getSubDotVal = [&](Value val, int64_t mm, int64_t kk, int64_t mStep,
                          int64_t kStep, bool hack = false) {
    auto [shape, subType, subSize] = getSubTypeAndShape(val.getType(), hack);
    unsigned subIdx =
        (kk / subSize[1]) * (shape[0] / subSize[0]) + mm / subSize[0];
    Value subVal = b.create<ttgi::ExtractOp>(loc, subType, val, subIdx);
    auto subDotType = RankedTensorType::get(
        {mStep, kStep},
        val.getType().cast<RankedTensorType>().getElementType());
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
        Value subDotA = getSubDotVal(dot.getA(), mm, kk, dotShape.m, dotShape.k,
                                     tmpSet.count(dot));
        Value subDotB =
            getSubDotVal(dot.getB(), kk, nn, dotShape.k, dotShape.n);
        subDotC = b.create<tt::DotOp>(loc, subDotA, subDotB, subDotC,
                                      dot.getInputPrecisionAttr(),
                                      dot.getMaxNumImpreciseAccAttr());
        // hack for attention
        subDotC.getDefiningOp()->setAttr(
            "schedule-group",
            b.getIntegerAttr(b.getI32Type(), nn / dotShape.n));
      }
      subCs.push_back(subDotC);
    }
  }

  dot->replaceAllUsesWith(
      b.create<ttgi::GlueOp>(loc, dot.getType(), subCs)->getResults());
  dot->erase();
}

void MatchTargetSizePass::transformBroadcastOp(tt::BroadcastOp op) {
  OpBuilder b(op);
  auto loc = op->getLoc();
  RankedTensorType resType = op.getResult().getType();
  auto [shape, subType, subSize] = getSubTypeAndShape(resType);
  auto tType = cast<RankedTensorType>(subType);
  RankedTensorType srcType = op.getSrc().getType();
  unsigned srcDim0 = srcType.getShape()[0];
  unsigned dstDim0 = tType.getShape()[0];
  if (srcDim0 == dstDim0) {
    Value newOp = b.create<tt::BroadcastOp>(loc, tType, op.getSrc());
    unsigned num = resType.getShape()[1] / tType.getShape()[1];
    SmallVector<Value> ops(num, newOp);
    auto glue = b.create<ttgi::GlueOp>(loc, resType, ops);
    op->replaceAllUsesWith(glue->getResults());
    op->erase();
  } else {
    assert(srcDim0 == 2 * dstDim0);
    auto newTy = RankedTensorType::get({srcDim0, tType.getShape()[1]},
                                       tType.getElementType());
    auto newOp = b.create<tt::BroadcastOp>(loc, newTy, op.getSrc());
    auto extract0 = b.create<ttgi::ExtractOp>(loc, tType, newOp, 0);
    auto extract1 = b.create<ttgi::ExtractOp>(loc, tType, newOp, 1);
    SmallVector<Value> ops{extract0, extract1, extract0, extract1,
                           extract0, extract1, extract0, extract1};
    auto glue = b.create<ttgi::GlueOp>(loc, resType, ops);
    op->replaceAllUsesWith(glue->getResults());
    op->erase();
  }
  return;
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

  // hack for attn
  bool hack = false;
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    auto ptrTy = store.getPtr().getType();
    if (cast<tt::PointerType>(ptrTy).getAddressSpace() == 3) {
      hack = true;
    }
  }
  if (auto load = dyn_cast<tt::LoadOp>(op)) {
    auto ptrTy = load.getPtr().getType();
    if (cast<tt::PointerType>(ptrTy).getAddressSpace() == 3) {
      hack = true;
    }
  }
  auto [shape, subType, subSize] = getSubTypeAndShape(type, hack);

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
        llvm::transform(
            op->getOperands(), std::back_inserter(newOperands),
            [&](Value operand) {
              Type type = operand.getType();
              if (isa<tt::BroadcastOp>(op))
                return operand;
              else if (isa<tt::PointerType, RankedTensorType>(type)) {
                Type subOpndType = std::get<1>(getSubTypeAndShape(type, hack));
                Value newOp =
                    b.create<ttgi::ExtractOp>(loc, subOpndType, operand, idx);
                return newOp;
              } else
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
