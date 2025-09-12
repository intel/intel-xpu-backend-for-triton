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
///   %tile_a0 = ttig.extract %a[0] : tensor<32x32xf16>
///            -> tensor<8x16xf16>
///   %tile_b0 = ttig.extract %b[0] : tensor<32x32xf16>
///            -> tensor<16x16xf16>
///   %dot_0 = tt.dot %tile_a0, %tile_b0 : tensor<8x16xf16>, tensor<16x16xf16>
///          -> tensor<8x16xf32>
///   %tile_a4 = ttig.extract %a[4] : tensor<32x32xf16>
///            -> tensor<8x16xf16>
///   %tile_b1 = ttig.extract %b[1] : tensor<32x32xf16>
///            -> tensor<16x16xf16>
///   %dot_1 = tt.dot %tile_a4, %tile_b1 : tensor<8x16xf16>, tensor<16x16xf16>
///          -> tensor<8x16xf32>
///   ...
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Dialect/TritonIntelGPU/IR/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
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

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonintelgpu-match-target-size"

namespace {

// Encode the native operation sizes supported by the target architecture.
class TargetArchNativeSizes {
public:
  struct DotShape {
    DotShape(unsigned m, unsigned n, unsigned k) : m(m), n(n), k(k) {
      assert(m != 0 && n != 0 && k != 0 && "expecting valid shape");
    }

    const unsigned m;
    const unsigned n;
    const unsigned k;
  };

  struct BlockMemShape {
    BlockMemShape(unsigned rowsA, unsigned columnsA, unsigned rowsB,
                  unsigned columnsB)
        : rowsA(rowsA), columnsA(columnsA), rowsB(rowsB), columnsB(columnsB) {
      assert(rowsA != 0 && columnsA != 0 && rowsB != 0 && columnsB != 0 &&
             "expecting valid shape");
    }

    const unsigned rowsA;
    const unsigned columnsA;
    const unsigned rowsB;
    const unsigned columnsB;
  };

  TargetArchNativeSizes() = default;

  void setDotShape(unsigned bitWidth, DotShape &&shape) {
    assert(!dotShapes.contains(bitWidth) && "Dot shape already set");
    dotShapes.try_emplace(bitWidth, std::move(shape));
  }
  void setBlockMemShape(unsigned bitWidth, BlockMemShape &&shape) {
    assert(!blockMemShapes.contains(bitWidth) &&
           "Block memory access shape already set");
    blockMemShapes.try_emplace(bitWidth, std::move(shape));
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

static tt::LoadOp findUsedLoad(Value val) {
  tt::LoadOp loadOp;
  auto defOp = val.getDefiningOp();
  if (isa_and_nonnull<tt::LoadOp>(defOp))
    return cast<tt::LoadOp>(defOp);
  for (auto u : defOp->getOperands()) {
    if (tt::LoadOp ld = findUsedLoad(u)) {
      loadOp = ld;
      break;
    }
  }
  return loadOp;
}

static bool getTransposeFlagFromValue(Value val) {
  Value loadPtr = val;
  // backward: from dot operands to tt.load
  if (llvm::any_of(val.getUsers(),
                   [](Operation *user) { return isa<tt::DotOp>(user); })) {
    // is a dot operand
    tt::LoadOp load = findUsedLoad(val);
    if (load)
      loadPtr = load.getPtr();
  }

  if (auto extractOp = loadPtr.getDefiningOp<ttgi::ExtractOp>()) {
    loadPtr = extractOp.getBase();
  }
  // forward: from tt.load/tt.advance to dot
  if (auto blockArg = dyn_cast<BlockArgument>(loadPtr)) {
    unsigned argIdx = blockArg.getArgNumber();
    if (auto loopLikeOp = dyn_cast<LoopLikeOpInterface>(
            blockArg.getParentBlock()->getParentOp()))
      loadPtr = loopLikeOp.getInits()[argIdx - 1];
  }

  if (auto glueOp = loadPtr.getDefiningOp<ttgi::GlueOp>()) {
    if (isa_and_present<tt::MakeTensorPtrOp, tt::AdvanceOp>(
            glueOp.getOperands()[0].getDefiningOp()))
      loadPtr = glueOp.getOperands()[0];
  }

  if (auto tensorPtr = loadPtr.getDefiningOp<tt::MakeTensorPtrOp>()) {
    ArrayRef<int32_t> order = tensorPtr.getOrder();
    auto rank = order.size();
    return (order[rank - 2] != 1);
  }

  if (auto advOp = loadPtr.getDefiningOp<tt::AdvanceOp>())
    return getTransposeFlagFromValue(advOp.getPtr());

  return false;
}

static void rewriteLoadWithSLM(ModuleOp &m, DenseSet<Value> &dotWithSLMOperands,
                               MLIRContext *ctx) {
  // FIXME: Use a cost model to decide when to do rewrite
  dotWithSLMOperands.clear();
  auto funcsVec = llvm::to_vector(m.getBody()->getOps<tt::FuncOp>());
  if (funcsVec.size() == 0)
    return;

  auto loadsVec = llvm::to_vector(funcsVec[0].getBody().getOps<tt::LoadOp>());
  if (loadsVec.size() == 0)
    return;
  tt::LoadOp load = loadsVec[0];

  // load is not in a loop-like op
  if (load->getParentOfType<LoopLikeOpInterface>())
    return;

  auto dot = cast<tt::DotOp>(*load->getUsers().begin());
  // load is not used by tt.dot
  if (llvm::none_of(load->getUsers(),
                    [&](Operation *user) { return isa<tt::DotOp>(user); }))
    return;

  auto loopOp = dyn_cast_or_null<LoopLikeOpInterface>(dot->getParentOp());
  // skip dot which is not in a loop
  if (!loopOp)
    return;

  Location loc = load.getLoc();
  OpBuilder b(load);
  auto type = cast<RankedTensorType>(load.getType());
  unsigned bytes = type.getNumElements() * type.getElementTypeBitWidth() / 8;
  unsigned numWarps = ttg::lookupNumWarps(load);
  unsigned slmSize = numWarps * bytes;

  // TODO: use LocalAllocOp for SLM allocation
  static constexpr char sharedAttr[] = "ttg.shared";
  m->setAttr(sharedAttr,
             mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32), slmSize));
  auto func = load->getParentOfType<FunctionOpInterface>();

  auto ptrToSharedMemTy = tt::PointerType::get(
      type.getElementType(), TritonGEN::TritonGENMemorySpace::kWorkgroup);
  LogicalResult inserted = func.insertArgument(
      func.getNumArguments(), ptrToSharedMemTy, {}, func.getLoc());
  assert(succeeded(inserted) && "failed to insert argument");
  b.setInsertionPointAfter(load);
  auto subgroupId =
      b.create<mlir::gpu::SubgroupIdOp>(loc, /*upperBound=*/nullptr);
  auto warpId = b.create<arith::IndexCastOp>(loc, b.getI32Type(), subgroupId);
  // hardcode: we use i16ptr here, so bytes / 2
  auto warpSize = b.create<arith::ConstantIntOp>(loc, bytes / 2, 32);
  auto offset = b.create<arith::MulIOp>(loc, warpId, warpSize);
  auto block = load->getBlock();
  auto arg = block->getArgument(block->getNumArguments() - 1);
  auto base = b.create<tt::AddPtrOp>(loc, ptrToSharedMemTy, arg, offset);
  SmallVector<Value> shape;
  shape.push_back(b.create<arith::ConstantIntOp>(loc, type.getShape()[0], 64));
  shape.push_back(b.create<arith::ConstantIntOp>(loc, type.getShape()[1], 64));
  SmallVector<Value> strides;
  strides.push_back(
      b.create<arith::ConstantIntOp>(loc, type.getShape()[1], 64));
  strides.push_back(b.create<arith::ConstantIntOp>(loc, 1, 64));
  SmallVector<Value> offsets;
  offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
  offsets.push_back(b.create<arith::ConstantIntOp>(loc, 0, 32));
  auto loadPtrTy = cast<tt::PointerType>(load.getPtr().getType());
  auto ptrTy = tt::PointerType::get(loadPtrTy.getPointeeType(), 3);
  auto ptr =
      b.create<tt::MakeTensorPtrOp>(loc, ptrTy, base, shape, strides, offsets,
                                    b.getDenseI32ArrayAttr({1, 0}));
  auto store = b.create<tt::StoreOp>(loc, ptr, load, tt::CacheModifier::NONE,
                                     tt::EvictionPolicy::NORMAL);

  b.setInsertionPoint(dot);
  auto newLoad =
      b.create<tt::LoadOp>(dot.getLoc(), ptr, tt::CacheModifier::NONE,
                           tt::EvictionPolicy::NORMAL, false);
  Value res = load.getResult();
  dotWithSLMOperands.insert(dot);
  res.replaceAllUsesExcept(newLoad, store);
}
class MatchTargetSizePass
    : public triton::gpu::intel::impl::TritonIntelGPUMatchTargetSizeBase<
          MatchTargetSizePass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp m = getOperation();

    // By default, tritongpu are lowered to simt mode (threads-per-warp=16)
    // instead of simd mode (threads-per-warp=1).
    // FIXME: force threads-per-warp=16 in simt(this should be done via an
    // analysis designed to determine whether the kernel contains tt.dot
    // operations that use block pointers).
    m->setAttr("ttg.threads-per-warp",
               IntegerAttr::get(IntegerType::get(ctx, 32), 16));

    Workload workload = Workload::None;
    m.walk([&](scf::ForOp forOp) {
      if (Attribute attr = forOp->getAttr(AttrWorkloadName))
        workload = static_cast<Workload>(cast<IntegerAttr>(attr).getInt());
    });

    initNativeOperationSizes(workload);

    // this is ad-hoc for flash attention load/store Q on SLM
    if (tools::getBoolEnv("TRITON_INTEL_ENABLE_FIRST_LOAD_TO_SLM"))
      rewriteLoadWithSLM(m, dotWithSLMOperands, ctx);

    // Collect the result layout of "interesting" `tt.dot` operations.
    // A candidate 'tt.dot' operation yields a tensor with a warp layout.
    m.walk([&](tt::DotOp dot) {
      auto resultType = cast<RankedTensorType>(dot.getResult().getType());
      if (isCandidate(resultType))
        dotAttrs.insert(resultType.getEncoding());
    });

    auto hasSliceAttr = [](Type type) {
      auto tType = dyn_cast<RankedTensorType>(type);
      return tType && isa<ttg::SliceEncodingAttr>(tType.getEncoding());
    };

    // Split operations to match the target architecture native shapes.
    m.walk<WalkOrder::PreOrder>([&](Operation *op) {
      SmallVector<Type> types(op->getOperandTypes().begin(),
                              op->getOperandTypes().end());
      SmallVector<Type> resultTypes(op->getResultTypes().begin(),
                                    op->getResultTypes().end());
      types.append(resultTypes);

      if (auto mr = dyn_cast<tt::MakeRangeOp>(op)) {
        // FIXME: MakeRange's rank-1 tensors are not candidates in the original
        // sense of the advance path, but we need to split them here anyways.
        transformMakeRangeOp(mr);
        return WalkResult::advance();
      }

      if (llvm::none_of(types, [this](Type type) { return isCandidate(type); }))
        return WalkResult::advance();
      if (isa<scf::ForOp, scf::YieldOp>(op))
        return WalkResult::advance();
      if (auto expand = dyn_cast<tt::ExpandDimsOp>(op))
        return WalkResult::advance();

      // FIXME: hack it for now
      if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op))
        convert.getResult().replaceAllUsesWith(convert.getSrc());
      else if (auto reduce = dyn_cast<tt::ReduceOp>(op))
        transformReduceOp(reduce);
      else if (op->getNumResults() == 1 &&
               hasSliceAttr(op->getResultTypes()[0]))
        return WalkResult::advance();
      else if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
        recordRootSubSize(op);
        transformArithConstantOp(cstOp);
      } else if (auto ptrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
        recordRootSubSize(op);
        transformMakeTensorPtrOp(ptrOp);
      } else if (auto dot = dyn_cast<tt::DotOp>(op))
        transformDotOp(dot);
      else if (auto bc = dyn_cast<ttgi::BroadcastOp>(op))
        transformBroadcastOp(bc);
      else
        transformGenericOp(op);

      return WalkResult::advance();
    });

    LLVM_DEBUG(llvm::dbgs() << "Canonicalizing...\n");
    LLVM_DEBUG(llvm::dbgs() << "Module before canonicalization:\n"
                            << m << "\n\n");
    canonicalize();
    LLVM_DEBUG(llvm::dbgs() << "Module after canonicalization:\n"
                            << m << "\n\n");
  }

private:
  /// Initialize the native operation sizes supported by the target
  /// architecture.
  void initNativeOperationSizes(Workload workload = Workload::None);

  /// Determine whether the given type is a tensor (or a pointer to a tensor)
  /// that has a warp layout or a dot layout with a parent warp layout.
  bool isCandidate(Type type) const;

  /// Canonicalize operations (e.g. remove redundant tt.extract, tt.glue)
  void canonicalize();

  void recordRootSubSize(Operation *op);
  SmallVector<int64_t> getSubOpSize(RankedTensorType op,
                                    bool isTransposed) const;

  std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
  getSubTypeAndShape(Type type, bool isTransposed = false, bool useSLM = false,
                     bool keepEncoding = false) const;

  Value getSubVal(Operation *op, Value val, ArrayRef<int64_t> srcOffset,
                  ArrayRef<int64_t> dstSize);

  /// Transformations for specific operations.
  void transformMakeTensorPtrOp(tt::MakeTensorPtrOp op);
  void transformArithConstantOp(arith::ConstantOp op);
  void transformDotOp(tt::DotOp dot);
  void transformReduceOp(tt::ReduceOp op);
  void transformTransposedReduceOp(tt::ReduceOp op);
  void transformBroadcastOp(ttgi::BroadcastOp op);
  void transformMakeRangeOp(tt::MakeRangeOp op);

  /// Generic transformation.
  void transformGenericOp(Operation *op);

  /// Record the native size supported by the target implementation.
  DenseMap<Attribute, SmallVector<int64_t>> sizePerAttrMap;

  /// Record the native size supported by the target implementation for
  /// transposed type.
  DenseMap<Attribute, SmallVector<int64_t>> sizePerAttrMapTransposed;

  /// Collects the result layout of the `tt.dot` operations in the module.
  DenseSet<Attribute> dotAttrs;

  /// The native operation sizes supported by the target architecture.
  TargetArchNativeSizes nativeSizes;

  /// Dot ops that use operands from SLM(outer loop)
  DenseSet<Value> dotWithSLMOperands;
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
///   %glue = ttig.glue %a, %b : tensor<4x4xf32>, tensor<4x4xf32>
///         -> tensor<8x4xf32>
///   scf.for %i = %lb to %ub step %step (%arg = %glue) {
///     %extract = ttig.extract %arg[0] : tensor<8x4xf32>
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

    rewriter.setInsertionPointAfter(newForOp);

    idx = 0;
    for (auto [result, init] :
         llvm::zip(forOp.getResults(), forOp.getInits())) {
      Operation *definingOp = init.getDefiningOp();

      // Loop-carried value was not split by this pattern, just rewire all users
      // to the new scf.for operation.
      if (!isa_and_nonnull<ttgi::GlueOp>(definingOp)) {
        result.replaceAllUsesWith(newForOp.getResults()[idx]);
        ++idx;
        continue;
      }

      // Re-glue individual results together _after_ the loop. This enables
      // canonicalization of extract ops and dependent loops.
      auto glue = cast<ttgi::GlueOp>(definingOp);
      auto reglue = rewriter.create<ttgi::GlueOp>(
          forOp->getLoc(), result.getType(),
          newForOp->getResults().slice(idx, glue.getOperands().size()));
      result.replaceAllUsesWith(reglue);
      idx += glue->getOperands().size();
    }

    for (Operation *deleteOp : deleteList)
      rewriter.eraseOp(deleteOp);

    rewriter.eraseOp(forOp);

    return success();
  }

  bool isCandidateLoop(scf::ForOp forOp) const {
    // If none of the loop init values is defined by a 'glue' operation there is
    // nothing to do.
    if (llvm::none_of(forOp.getInits(), [](Value init) {
          return init.getDefiningOp<ttgi::GlueOp>();
        })) {
      return false;
    }

    // Bail out if any user of a 'glue' init value is not an 'extract'
    // operation.
    for (auto [arg, init] :
         llvm::zip(forOp.getRegionIterArgs(), forOp.getInits())) {
      if (!init.getDefiningOp<ttgi::GlueOp>())
        continue;

      if (llvm::any_of(arg.getUsers(), [](Operation *user) {
            return !isa<ttgi::ExtractOp>(user);
          }))
        return false;
    }

    // Bail out if the loop result is not used by an 'extract' operation, or
    // another loop.
    if (forOp->getNumResults() == 1 &&
        llvm::any_of(forOp.getResult(0).getUsers(), [](Operation *user) {
          return !isa<ttgi::ExtractOp, scf::ForOp>(user);
        }))
      return false;

    return true;
  }
};

void MatchTargetSizePass::initNativeOperationSizes(Workload workload) {
  // FIXME: sets the target dot shape natively supported by the target
  // architecture using the target architecture information when available.
  // These values works for PVC.

  nativeSizes.setDotShape(8, {8, 16, 32});
  nativeSizes.setDotShape(16, {8, 16, 16});
  nativeSizes.setDotShape(32, {8, 16, 8});

  nativeSizes.setBlockMemShape(8, {16, 64, 32, 32});
  if (workload == Workload::Attention)
    nativeSizes.setBlockMemShape(16, {32, 32, 32, 16});
  else
    nativeSizes.setBlockMemShape(16, {32, 32, 32, 32});

  nativeSizes.setBlockMemShape(32, {8, 8, 8, 16});

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

  if (failed(applyPatternsGreedily(m, std::move(patterns))))
    signalPassFailure();
}

void MatchTargetSizePass::recordRootSubSize(Operation *op) {
  Type type;
  bool isTransposed = false;
  if (auto cstOp = dyn_cast<arith::ConstantOp>(op)) {
    type = cstOp.getResult().getType();
  } else if (auto tensorPtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
    type = tensorPtrOp.getResult().getType();
    ArrayRef<int32_t> order = tensorPtrOp.getOrder();
    auto rank = order.size();
    isTransposed = (order[rank - 2] != 1);
  }

  auto ptrType = dyn_cast<tt::PointerType>(type);
  // extract until a non-PointerType
  while (isa_and_nonnull<tt::PointerType>(ptrType)) {
    type = ptrType.getPointeeType();
    ptrType = dyn_cast<tt::PointerType>(type);
  }

  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Attribute layout = tensorType.getEncoding();
    assert(layout && "Expecting a valid layout");
    if (isTransposed && sizePerAttrMapTransposed.count(layout) == 0) {
      sizePerAttrMapTransposed[layout] = getSubOpSize(tensorType, isTransposed);
    }
    if (!isTransposed && sizePerAttrMap.count(layout) == 0) {
      sizePerAttrMap[layout] = getSubOpSize(tensorType, isTransposed);
    }
    return;
  }
}

/// Return the native size supported by the target architecture.
SmallVector<int64_t>
MatchTargetSizePass::getSubOpSize(RankedTensorType type,
                                  bool isTransposed) const {
  Attribute layout = type.getEncoding();
  assert(layout && "Expecting a valid layout");

  // Dot related operation.
  const TargetArchNativeSizes::DotShape &dotShape =
      nativeSizes.getDotShape(type.getElementTypeBitWidth());
  if (dotAttrs.count(layout)) {
    return {dotShape.m, dotShape.n};
  } else if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout)) {
    if (isTransposed && dotAttr.getOpIdx() == 1)
      return {dotShape.k, dotShape.n};
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
      // max 2d block prefetch width is 16 for 32-bit datatype
      subSize[1] = std::min<int64_t>(sizeInBits == 32 ? 16 : 32, shape[1]);
      // max 2d block load height is 32
      subSize[0] = std::min<int64_t>(32, shape[0]);
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

/// FIXME: add a map for look up
/// return [shape, subType, subSize] for a tensor (or pointer to tensor)
std::tuple<SmallVector<int64_t>, Type, SmallVector<int64_t>>
MatchTargetSizePass::getSubTypeAndShape(Type type, bool isTransposed,
                                        bool useSLM, bool keepEncoding) const {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    Attribute layout = tensorType.getEncoding();
    assert(layout && "Expecting a valid layout");
    SmallVector<int64_t> shape = to_vector(tensorType.getShape());
    SmallVector<int64_t> subSize = isTransposed
                                       ? sizePerAttrMapTransposed.at(layout)
                                       : sizePerAttrMap.at(layout);
    // specific for flash attention
    if (useSLM) {
      subSize[0] = 16;
      subSize[1] = 64;
    } else {
      // Never exceed the shape of the original type.
      subSize[0] = std::min(subSize[0], shape[0]);
      subSize[1] = std::min(subSize[1], shape[1]);
    }

    auto subType = RankedTensorType::get(subSize, tensorType.getElementType(),
                                         keepEncoding ? tensorType.getEncoding()
                                                      : Attribute{});
    return {shape, subType, subSize};
  }

  if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
    Type pointeeType = ptrType.getPointeeType();
    auto [shape, subType, subSize] =
        getSubTypeAndShape(pointeeType, isTransposed, useSLM, keepEncoding);
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
  constexpr StringLiteral SharedAttrName = "ttg.shared";
  if (!m->getAttr(SharedAttrName)) {
    m->setAttr(SharedAttrName, b.getIndexAttr(size));
    LogicalResult inserted = func.insertArgument(func.getNumArguments(), ptrTy,
                                                 b.getDictionaryAttr({}), loc);
    assert(succeeded(inserted) && "failed to insert argument");
  }
  return func.getArguments().back();
}

static SmallVector<Value> glueForReduction(OpBuilder &builder, Location loc,
                                           ArrayRef<Value> subVals) {
  assert(subVals.size() % 2 == 0 && "Expecting even number of values");
  SmallVector<Value> result;
  SmallVector<Value> lhs;
  SmallVector<Value> rhs;
  for (auto [index, value] : llvm::enumerate(subVals))
    (index % 2 == 0 ? lhs : rhs).push_back(value);
  auto operandType = cast<RankedTensorType>(subVals.front().getType());
  SmallVector<int64_t> glueShape(operandType.getShape());
  assert(glueShape.size() == 2 && "Expecting two-dimensional operator");
  glueShape[0] *= 2;
  RankedTensorType glueType =
      RankedTensorType::get(glueShape, operandType.getElementType());
  llvm::transform(llvm::zip_equal(lhs, rhs), std::back_inserter(result),
                  [&builder, loc, glueType](auto pair) -> Value {
                    auto [lhs, rhs] = pair;
                    return builder.create<triton::gpu::intel::GlueOp>(
                        loc, glueType, ValueRange{lhs, rhs});
                  });
  return result;
}

static Value allocateSLMForTransposedReduction(tt::ReduceOp op, unsigned step,
                                               OpBuilder &b) {
  Value src = op.getSrcs().front();
  auto srcTy = cast<RankedTensorType>(src.getType());
  Location loc = op.getLoc();

  // Fixed size for num_warps matrices of sg_size^2 shape.
  int64_t size = static_cast<int64_t>(step) * step *
                 srcTy.getElementTypeBitWidth() / 8 * ttg::lookupNumWarps(op);
  Type allocTy = cast<RankedTensorType>(src.getType()).getElementType();
  Type ptrTy = tt::PointerType::get(allocTy, tt::TritonGEN::kWorkgroup);
  return hackAlloc(b, loc, ptrTy, size);
}

void MatchTargetSizePass::transformTransposedReduceOp(tt::ReduceOp op) {
  Location loc = op.getLoc();
  OpBuilder b(op);
  assert(op.getSrcs().size() == 1 && "only support one src");
  Value src = op.getSrcs().front();
  auto srcTy = cast<RankedTensorType>(src.getType());
  unsigned dims = srcTy.getShape().size();
  unsigned axis = op.getAxis();
  assert(axis == dims - 1 && "only support last axis");
  assert(dims <= 2 && "only support 1D/2D tensor");
  int64_t outer = dims == 2 ? srcTy.getShape()[0] : 1;

  SmallVector<Value> glueVals;

  // Fixed for transpose reduction.
  constexpr unsigned subGroupSize = 16;
  constexpr unsigned step = subGroupSize;
  Value localBuffer = allocateSLMForTransposedReduction(op, subGroupSize, b);
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
    Type subType = dstType;
    auto combine = op.getCombineOp().front().getOperations().begin();
    StringAttr id = combine->getName().getIdentifier();
    Value acc;
    switch (subVals.size()) {
    case 1:
      acc = subVals[0];
      break;
    case 2: {
      Operation *acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      acc = acc01->getResult(0);
      break;
    }
    case 4: {
      Operation *acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      Operation *acc23 = b.create(loc, id, {subVals[2], subVals[3]}, subType);
      Operation *accOp = b.create(
          loc, id, {acc01->getResult(0), acc23->getResult(0)}, subType);
      acc = accOp->getResult(0);
      break;
    }
    default:
      assert(false && "add more reduce size support");
    }

    Value accT = b.create<ttgi::SubGroupTransposeOp>(loc, acc.getType(),
                                                     localBuffer, acc);
    auto sgReduce = b.create<tt::ReduceOp>(loc, accT, axis);
    Region &sgReduceRegion = sgReduce.getCombineOp();
    b.cloneRegionBefore(op.getCombineOp(), sgReduceRegion,
                        sgReduceRegion.end());
    glueVals.push_back(sgReduce->getResult(0));
  }

  auto resultType = cast<RankedTensorType>(op.getResultTypes()[0]);
  RankedTensorType glueType =
      RankedTensorType::get(resultType.getShape(), resultType.getElementType());
  Value glue = glueVals.size() == 1
                   ? glueVals.front()
                   : b.create<ttgi::GlueOp>(loc, glueType, glueVals);
  Value res = b.create<ttg::ConvertLayoutOp>(loc, resultType, glue);
  op->replaceAllUsesWith(ValueRange{res});
  op->erase();
}

void MatchTargetSizePass::transformReduceOp(tt::ReduceOp op) {
  if (mlir::triton::gpu::intel::applyTransposedReduction()) {
    transformTransposedReduceOp(op);
    return;
  }

  Location loc = op.getLoc();
  OpBuilder b(op);
  assert(op.getSrcs().size() == 1 && "only support one src");
  Value src = op.getSrcs().front();
  auto srcTy = cast<RankedTensorType>(src.getType());
  unsigned dims = srcTy.getShape().size();
  unsigned axis = op.getAxis();
  assert(axis == dims - 1 && "only support last axis");
  assert(dims <= 2 && "only support 1D/2D tensor");
  int64_t outer = dims == 2 ? srcTy.getShape()[0] : 1;

  SmallVector<Value> glueVals;
  unsigned step = 8; // FIXME: fixed to 8 for now.
  for (unsigned i = 0; i < outer; i += step) {
    SmallVector<Value> subVals;
    // FIXME: 16 is the supported IR reduce length
    for (unsigned j = 0; j < srcTy.getShape()[axis]; j += 16) {
      Value subVal = getSubVal(op, src, {i, j}, {step, 16});
      subVals.push_back(subVal);
    }
    auto subType = RankedTensorType::get({step, 16}, srcTy.getElementType());
    auto combine = op.getCombineOp().front().getOperations().begin();
    StringAttr id = combine->getName().getIdentifier();
    Value acc;
    switch (subVals.size()) {
    case 1:
      acc = subVals[0];
      break;
    case 2: {
      Operation *acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      acc = acc01->getResult(0);
      break;
    }
    case 4: {
      Operation *acc01 = b.create(loc, id, {subVals[0], subVals[1]}, subType);
      Operation *acc23 = b.create(loc, id, {subVals[2], subVals[3]}, subType);
      Operation *accOp = b.create(
          loc, id, {acc01->getResult(0), acc23->getResult(0)}, subType);
      acc = accOp->getResult(0);
      break;
    }
    default:
      assert(false && "add more reduce size support");
    }

    SmallVector<Value> subOps;
    for (unsigned j = 0; j < step; j++) {
      auto subType = RankedTensorType::get(16, srcTy.getElementType());
      Value subAcc = b.create<ttgi::ExtractOp>(loc, subType, acc, j);
      auto subRed = b.create<tt::ReduceOp>(loc, subAcc, 0);
      Region &subRegion = subRed.getCombineOp();
      b.cloneRegionBefore(op.getCombineOp(), subRegion, subRegion.end());
      subOps.push_back(subRed.getResult()[0]);
    }
    glueVals.append(subOps);
  }

  auto glue = b.create<ttgi::GlueOp>(loc, op.getResultTypes()[0], glueVals);
  op->replaceAllUsesWith(glue->getResults());
  op->erase();

  return;
}

void MatchTargetSizePass::transformMakeTensorPtrOp(tt::MakeTensorPtrOp op) {
  Type resultType = op.getResult().getType();
  ArrayRef<int32_t> order = op.getOrder();
  auto rank = order.size();
  bool isTransposed = (order[rank - 2] != 1);
  bool useSLM = (cast<tt::PointerType>(resultType).getAddressSpace() ==
                 TritonGEN::TritonGENMemorySpace::kWorkgroup);
  auto [shape, subType, subSize] =
      getSubTypeAndShape(resultType, isTransposed, useSLM);

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
        subOp.getResult().setType(dyn_cast<tt::PointerType>(subType));
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
                          int64_t kStep, bool useSLM = false) {
    bool isTransposed = false;
    if (auto dotEnc = dyn_cast<DotOperandEncodingAttr>(
            cast<RankedTensorType>(val.getType()).getEncoding())) {
      // FIXME: Just transposed B supported now
      if (dotEnc.getOpIdx() == 1) {
        isTransposed = getTransposeFlagFromValue(val);
      }
    }
    auto [shape, subType, subSize] =
        getSubTypeAndShape(val.getType(), isTransposed, useSLM);
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
        Value subDotA = getSubDotVal(dot.getA(), mm, kk, dotShape.m, dotShape.k,
                                     dotWithSLMOperands.count(dot));
        Value subDotB =
            getSubDotVal(dot.getB(), kk, nn, dotShape.k, dotShape.n);
        subDotC = b.create<tt::DotOp>(loc, subDotA, subDotB, subDotC,
                                      dot.getInputPrecisionAttr(),
                                      dot.getMaxNumImpreciseAccAttr());
        // hack for attention
        if (triton::tools::getBoolEnv("TRITON_INTEL_ENABLE_INSTR_SCHED"))
          if (auto definingOp = subDotC.getDefiningOp())
            definingOp->setAttr(
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

void MatchTargetSizePass::transformBroadcastOp(ttgi::BroadcastOp op) {
  OpBuilder b(op);
  Location loc = op->getLoc();
  RankedTensorType resType = op.getResult().getType();
  auto [shape, subType, subSize] = getSubTypeAndShape(resType);
  auto tType = cast<RankedTensorType>(subType);
  RankedTensorType srcType = op.getSrc().getType();
  unsigned srcDim0 = srcType.getShape()[0];
  unsigned dstDim0 = tType.getShape()[0];
  unsigned resDim0 = resType.getShape()[0];
  unsigned srcDim1 = srcType.getShape()[1];
  unsigned dstDim1 = tType.getShape()[1];
  unsigned resDim1 = resType.getShape()[1];
  Operation *glue;
  if (srcDim0 == dstDim0) {
    Value newOp = b.create<ttgi::BroadcastOp>(loc, tType, op.getSrc());
    unsigned num = resType.getShape()[1] / tType.getShape()[1];
    SmallVector<Value> ops(num, newOp);
    glue = b.create<ttgi::GlueOp>(loc, resType, ops);
  } else if (srcDim0 == 2 * dstDim0) {
    auto newTy = RankedTensorType::get({srcDim0, tType.getShape()[1]},
                                       tType.getElementType());
    auto newOp = b.create<ttgi::BroadcastOp>(loc, newTy, op.getSrc());
    auto extract0 = b.create<ttgi::ExtractOp>(loc, tType, newOp, 0);
    auto extract1 = b.create<ttgi::ExtractOp>(loc, tType, newOp, 1);
    SmallVector<Value> ops{extract0, extract1, extract0, extract1,
                           extract0, extract1, extract0, extract1};
    glue = b.create<ttgi::GlueOp>(loc, resType, ops);
  } else if (srcDim0 == 1 && srcDim1 == resDim1) {
    // Handle row-vector broadcasts, e.g. 1x64 --> 16x64.
    // This kind of broadcast requires that the tensor type is kept intact by
    // SIMT lowering, hence propagate the encoding here.
    auto subRowVecTy =
        RankedTensorType::get({1, tType.getShape()[1]}, tType.getElementType(),
                              srcType.getEncoding());

    // How many extracts do we need to cover the width of the input tensor?
    unsigned nExtracts = srcDim1 / dstDim1;
    SmallVector<Value> subBroadcasts;
    for (int i = 0; i < nExtracts; ++i) {
      auto ext = b.create<ttgi::ExtractOp>(loc, subRowVecTy, op.getSrc(), i);
      auto sbc = b.create<ttgi::BroadcastOp>(loc, tType, ext);
      subBroadcasts.push_back(sbc);
    }

    // How often do we need to repeat a sub broadcast to cover the height of the
    // result tensor?
    unsigned nRepeats = resDim0 / dstDim0;
    SmallVector<Value> ops;
    for (int i = 0; i < nRepeats * nExtracts; ++i)
      ops.push_back(subBroadcasts[i / nRepeats]);

    glue = b.create<ttgi::GlueOp>(loc, resType, ops);
  } else {
    llvm::report_fatal_error("Unhandled broadcast");
  }

  op->replaceAllUsesWith(glue->getResults());
  op->erase();

  return;
}

void MatchTargetSizePass::transformMakeRangeOp(tt::MakeRangeOp op) {
  auto mod = op->getParentOfType<mlir::ModuleOp>();
  int subgroupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

  unsigned start = op.getStart();
  unsigned end = op.getEnd();
  assert(start == 0 && (end <= subgroupSize || end % subgroupSize == 0) &&
         "Unsupported range");

  if (end <= subgroupSize)
    // nothing to do
    return;

  // Transform the range like this: (SG = subgroup size = 16)
  // makeRange(0, N) = glue(
  //   splat(0 * SG) + makeRange(0, SG),
  //   splat(1 * SG) + makeRange(0, SG),
  //   ...
  //   splat((N/SG-1) * SG) + makeRange(0, SG)
  // )

  OpBuilder b(op);
  Location loc = op.getLoc();
  RankedTensorType origTy = op.getType();
  Type elemTy = origTy.getElementType();
  // Propagate encoding to keep tensor during SIMT lowering.
  auto subRangeTy =
      RankedTensorType::get({subgroupSize}, elemTy, origTy.getEncoding());
  auto subRange = b.create<tt::MakeRangeOp>(loc, subRangeTy, 0, subgroupSize);
  SmallVector<Value> subRanges;
  for (int i = 0; i < end / subgroupSize; ++i) {
    Value offset =
        b.create<arith::ConstantIntOp>(loc, elemTy, i * subgroupSize);
    Value offsetTensor = b.create<tt::SplatOp>(loc, subRangeTy, offset);
    subRanges.push_back(b.create<arith::AddIOp>(loc, subRange, offsetTensor));
  }
  auto glue = b.create<ttgi::GlueOp>(loc, op.getType(), subRanges);
  op.replaceAllUsesWith(glue->getResults()[0]);
  op->erase();
}

void MatchTargetSizePass::transformGenericOp(Operation *op) {
  unsigned numResults = op->getResults().size();
  unsigned dotIdx = 2;
  Type type;

  bool isTransposed = false;
  switch (numResults) {
  case 0:
    // prefetch/store
    type = op->getOperand(0).getType();
    break;
  case 1: {
    // arith/math/advanceOp/loadOp
    type = op->getResultTypes()[0];
    // mark tt.load/tt.advance for dot A/B
    if (isa<RankedTensorType, tt::PointerType>(type))
      if (isa<tt::LoadOp, tt::AdvanceOp>(op)) {
        RankedTensorType tensorType;
        if (isa<RankedTensorType>(type))
          tensorType = dyn_cast<RankedTensorType>(type);
        else
          tensorType = dyn_cast<RankedTensorType>(
              dyn_cast<tt::PointerType>(type).getPointeeType());
        Attribute layout = tensorType.getEncoding();
        assert(layout && "Expecting a valid layout");

        tt::LoadOp load = dyn_cast<tt::LoadOp>(op);
        tt::AdvanceOp advOp = dyn_cast<tt::AdvanceOp>(op);
        Value loadPtr;
        if (load)
          loadPtr = load.getPtr();
        else
          loadPtr = advOp.getPtr();

        isTransposed = getTransposeFlagFromValue(loadPtr);
        if (auto dotAttr = dyn_cast<ttg::DotOperandEncodingAttr>(layout))
          dotIdx = dotAttr.getOpIdx();
      }
  } break;
  default:
    llvm_unreachable("Unexpected operation");
  }

  // for attention SLM
  bool useSLM = false;
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    auto ptrAS =
        cast<tt::PointerType>(store.getPtr().getType()).getAddressSpace();
    useSLM = (ptrAS == TritonGEN::TritonGENMemorySpace::kWorkgroup);
  }
  if (auto load = dyn_cast<tt::LoadOp>(op)) {
    auto ptrAS =
        cast<tt::PointerType>(load.getPtr().getType()).getAddressSpace();
    useSLM = (ptrAS == TritonGEN::TritonGENMemorySpace::kWorkgroup);
  }

  // Keep encoding on certain tensors to leave them untouched during SIMT
  // lowering. Currently, this is required for "row vectors" (= `tensor<1xN>`).
  bool keepEncoding = false;
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    ArrayRef<int64_t> shape = tensorType.getShape();
    keepEncoding = shape.size() == 2 && shape[0] == 1 && shape[1] > 1;
  }
  auto [shape, subType, subSize] =
      getSubTypeAndShape(type, isTransposed, useSLM, keepEncoding);

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
                            Type subOpndType = std::get<1>(getSubTypeAndShape(
                                type, isTransposed, useSLM, keepEncoding));
                            Value newOp = b.create<ttgi::ExtractOp>(
                                loc, subOpndType, operand, idx);
                            return newOp;
                          }
                          return operand;
                        });

        Operation *subOp = nullptr;
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

  if (numResults == 1)
    op->replaceAllUsesWith(b.create<ttgi::GlueOp>(loc, type, subOps));

  op->erase();
}

} // namespace
