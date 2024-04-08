//===-------------- PrefetchBlock.cpp -  ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass to add prefetch operations for targets that
/// supports them. This pass looks for SCF loops containing a 'tt.load'
/// operation feeding the operand of a 'tt.dot' operation and injects prefetch
/// operations before the loop and in each loop iteration.
///
/// Note: this pass add a layout attribute to the newly created prefetch
/// operations.
///
/// Limitations:
///   - only blocked pointers are supported
///   - it is expected that the 'convert-triton-to-tritongpu-warp' pass is run
///     before this pass
///
/// Example:
///   scf.for
///     %load = tt.load %block_ptr
///     tt.dot %load
///     tt.advance %block_ptr
///
/// becomes:
///   %prefetch_ptr = tt.make_tensor_ptr
///   tt.prefetch %prefetch_ptr
///   tt.advance %prefetch_ptr
///   scf.for
///     %load = tt.load %block_ptr
///     tt.dot %load
///     tt.advance %block_ptr
///     tt.prefetch %prefetch_ptr
///     tt.advance %prefetch_ptr
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include <memory>
#include <optional>

namespace mlir {
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-prefetch-block"

namespace {

/// Returns true if \p val has a single user of the given type \p tparam T and
/// false otherwise.
template <typename T>
bool hasSingleUserOfKindInLoop(Value val, LoopLikeOpInterface loopLike) {
  return llvm::count_if(val.getUsers(), [&](auto user) {
           return isa<T>(user) && loopLike->isProperAncestor(user);
         }) == 1;
}

/// Get the first user of \p val with kind \tparam T or std::nullopt otherwise.
template <typename T> std::optional<T> getFirstUserOfKind(Value val) {
  for (auto user : val.getUsers())
    if (isa<T>(user))
      return cast<T>(user);
  return std::nullopt;
}

/// Annotate prefetch type with layout encoding.
Type annotatePrefetchType(Type type, unsigned numWarps) {
  auto ptrType = dyn_cast<tt::PointerType>(type);

  RankedTensorType tType;
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    tType = tensorType;
  else if (ptrType)
    tType = cast<RankedTensorType>(ptrType.getPointeeType());
  else
    llvm_unreachable("Unexpected type");

  ArrayRef<int64_t> shape = tType.getShape();
  assert(shape.size() == 2 && "Expecting a 2D shape");

  SmallVector<unsigned> sizePerWarp(2), warpsPerCTA(2);
  int64_t m = shape[0], n = shape[1];

  // typical numWarps 4, 8, 16, 32, 64
  // naive way to get warp distribute
  int64_t sizeX = n < 32 ? n : 32; // elementtype
  int64_t numWarpsX = n / sizeX;
  // auto root = std::sqrt(numWarps);
  // assert(n >= 16);
  // if (n / 16 <= root)
  //   numWarpsX = n / 16;
  // else if (n / 32 <= root)
  //   numWarpsX = n / 32;
  // else if (n / 64 <= root)
  //   numWarpsX = n / 64;
  // else
  //   numWarpsX = n / 128;
  warpsPerCTA[1] = numWarpsX;
  warpsPerCTA[0] = numWarps / warpsPerCTA[1];
  sizePerWarp[1] = n / warpsPerCTA[1];
  sizePerWarp[0] = m / warpsPerCTA[0];
  auto ctaLayout =
      ttg::CTALayoutAttr::get(type.getContext(), {1, 1}, {1, 1}, {1, 0});
  auto blockLayout = ttg::BlockedEncodingAttr::get(
      type.getContext(), sizePerWarp, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
  auto newType = RankedTensorType::get(tType.getShape(), tType.getElementType(),
                                       blockLayout);
  if (ptrType)
    return tt::PointerType::get(newType, ptrType.getAddressSpace());

  return newType;
}

class PrefetchBlockPass
    : public TritonIntelGPUPrefetchBlockBase<PrefetchBlockPass> {
public:
  /// Groups information for a candidate load.
  struct LoadInfo {
    LoadInfo(tt::AdvanceOp advance, SmallVector<Value> offsets,
             tt::MakeTensorPtrOp blockPtr)
        : advance(advance), offsets(offsets), blockPtr(blockPtr) {}
    LoadInfo(const LoadInfo &other)
        : advance(other.advance), offsets(other.offsets),
          blockPtr(other.blockPtr) {}

    tt::AdvanceOp getAdvance() const { return advance; }
    const SmallVector<Value> getOffsets() const { return offsets; }
    tt::MakeTensorPtrOp getBlockPtr() const { return blockPtr; }

  private:
    tt::AdvanceOp advance;        /// AdvanceOp using the blocked pointer
    SmallVector<Value> offsets;   /// Offsets used by the AdvanceOp
    tt::MakeTensorPtrOp blockPtr; /// Operation defining the blocked pointer
  };

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ModuleOp mod = getOperation();

    for (auto func : mod.getOps<tt::FuncOp>()) {
      // Collect candidate loads for each SCF for loop in this function.
      for (auto loop : func.getOps<scf::ForOp>()) {
        // FIXME: skip loop nests for now.
        if (!loop.getOps<LoopLikeOpInterface>().empty())
          continue;

        SmallVector<tt::LoadOp> loads;
        collectCandidatesLoadsInLoop(loop, loads);
        if (!loads.empty())
          loopLoads[loop].append(loads);
      }
      if (loopLoads.empty())
        continue;

      // Prefetch candidate loads collected.
      for (auto loop : func.getOps<scf::ForOp>())
        transformLoop(loop);
    }
  }

private:
  /// Collect candidate loads in \p loop.
  void collectCandidatesLoadsInLoop(scf::ForOp loop,
                                    SmallVectorImpl<tt::LoadOp> &loopLoads);

  /// Determines whether a load (in a loop) is a candidate. If it is, add an
  /// entry to the 'loadToLoadInfo' map and return true, otherwise return false.
  bool isCandidateLoad(tt::LoadOp load, scf::ForOp loop);

  /// Create a 'LoadInfo' object for the given \p load.
  std::optional<PrefetchBlockPass::LoadInfo>
  createLoadInfo(tt::LoadOp load, scf::ForOp loopLike) const;

  /// Find the defining 'MakeTensorPtrOp' operation for the given \p ptr.
  std::optional<tt::MakeTensorPtrOp>
  findDefiningMakeTensorPtrOp(scf::ForOp loop, Value ptr) const;

  /// Insert prefetching operation in the given \p loop.
  void transformLoop(scf::ForOp) const;

  /// Insert prefetch operations in the preheader of the given \p loop and
  /// return them in \p prefetchPtrs.
  void injectPrefetchOpsInPreheader(scf::ForOp loop,
                                    SmallVectorImpl<Value> &prefetchPtrs) const;

  /// Insert prefetch operations in the body of the given \p loop and return
  /// them in \p prefetchPtrs.
  void injectPrefetchOpsInBody(scf::ForOp loop,
                               SmallVectorImpl<Value> &prefetchPtrs) const;

  /// Map between a SCF loop and the candidate loads for the transformation.
  DenseMap<scf::ForOp, SmallVector<tt::LoadOp>> loopLoads;

  /// Map between a candidate load and its associate LoadInfo object.
  DenseMap<tt::LoadOp, LoadInfo> loadToLoadInfo;
};

void PrefetchBlockPass::collectCandidatesLoadsInLoop(
    scf::ForOp loop, SmallVectorImpl<tt::LoadOp> &loopLoads) {
  assert(loopLoads.empty() && "Expecting an empty vector");

  LLVM_DEBUG(llvm::dbgs() << "Attempting to collect candidate loads in loop:\n"
                          << loop << "\n\n");

  loop.walk([&](Operation *op) {
    if (auto load = dyn_cast<tt::LoadOp>(op)) {
      if (!isCandidateLoad(load, loop))
        return WalkResult::advance();

      loopLoads.push_back(load);
      LLVM_DEBUG(llvm::dbgs() << "Collected: " << load << "\n");
    }
    return WalkResult::advance();
  });

  LLVM_DEBUG(llvm::dbgs() << "Collected " << loopLoads.size() << " loads\n");
}

/// Determines whether a load (in a loop) is a candidate. A candidate load:
///   - must use a block pointer
///   - the block pointer must have 2 users in the loop, a 'tt.advance' and the
///     'tt.load' operation
///   - the result of the load must be used by a 'tt.dot' operation
///   - satisfy all conditions required in order to create a 'LoadInfo' object
///     for the load
///
/// FIXME: we assume the loop is an immediate parent of the load for now.
bool PrefetchBlockPass::isCandidateLoad(tt::LoadOp load, scf::ForOp loop) {
  Value ptr = load.getPtr();
  if (!isa<tt::PointerType>(ptr.getType()) || !loop->isProperAncestor(load))
    return false;

  unsigned numPtrUsers = range_size(ptr.getUsers());
  if (numPtrUsers != 2)
    return false;

  if (!hasSingleUserOfKindInLoop<tt::AdvanceOp>(ptr, loop) ||
      !hasSingleUserOfKindInLoop<tt::LoadOp>(ptr, loop))
    return false;

  Value loadRes = load.getResult();
  unsigned numLoadUsers = range_size(loadRes.getUsers());
  if (numLoadUsers != 1)
    return false;

  if (!hasSingleUserOfKindInLoop<tt::DotOp>(loadRes, loop))
    return false;

  std::optional<LoadInfo> loadInfo = createLoadInfo(load, loop);
  if (!loadInfo.has_value())
    return false;

  assert(!loadToLoadInfo.contains(load) && "Unexpected entry in the map");
  loadToLoadInfo.insert({load, *loadInfo});

  return true;
}

/// Create a LoadInfo for the given \p load if possible.
/// Notes:
///   - a 'tt.advance' operation must advance the load pointer using constant
///     offsets.
///   - the 'tt.MakeTensorPtrOp' operation must define the load pointer
std::optional<PrefetchBlockPass::LoadInfo>
PrefetchBlockPass::createLoadInfo(tt::LoadOp load, scf::ForOp loop) const {
  std::optional<tt::AdvanceOp> advance =
      getFirstUserOfKind<tt::AdvanceOp>(load.getPtr());
  if (!advance.has_value())
    return std::nullopt;

  SmallVector<OpFoldResult> rawOffsets = advance->getOffsets();
  if (!getConstantIntValues(rawOffsets).has_value())
    return std::nullopt;

  std::optional<tt::MakeTensorPtrOp> blockPtr =
      findDefiningMakeTensorPtrOp(loop, load.getPtr());
  if (!blockPtr.has_value())
    return std::nullopt;

  SmallVector<Value> offsets;
  llvm::transform(rawOffsets, std::back_inserter(offsets),
                  [](OpFoldResult ofr) { return cast<Value>(ofr); });

  LoadInfo loadInfo(*advance, offsets, *blockPtr);
  return loadInfo;
}

/// Transitively find the defining makeTensorPtrOp operation of \p ptr.
/// FIXME: extends to more complex patterns.
std::optional<tt::MakeTensorPtrOp>
PrefetchBlockPass::findDefiningMakeTensorPtrOp(scf::ForOp loop,
                                               Value ptr) const {
  if (auto arg = dyn_cast<BlockArgument>(ptr)) {
    auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
    return findDefiningMakeTensorPtrOp(loop, loopArg);
  }

  if (auto op = ptr.getDefiningOp()) {
    if (auto makePtrOp = dyn_cast<tt::MakeTensorPtrOp>(op))
      return makePtrOp;
  }

  return std::nullopt;
}

void PrefetchBlockPass::transformLoop(scf::ForOp loop) const {
  SmallVector<Value> prefetchPtrs;
  injectPrefetchOpsInPreheader(loop, prefetchPtrs);
  injectPrefetchOpsInBody(loop, prefetchPtrs);
}

/// Add prefetch operations in the loop pre-header.
void PrefetchBlockPass::injectPrefetchOpsInPreheader(
    scf::ForOp loop, SmallVectorImpl<Value> &prefetchPtrs) const {
  assert(prefetchPtrs.empty() && "Expecting an empty vector");

  ModuleOp mod = loop->getParentOfType<ModuleOp>();
  OpBuilder b(loop);

  for (tt::LoadOp load : loopLoads.at(loop)) {
    const LoadInfo &loadInfo = loadToLoadInfo.at(load);
    const unsigned numWarps = ttg::TritonGPUDialect::getNumWarps(mod);

    b.setInsertionPoint(loadInfo.getBlockPtr());
    auto ptr = cast<tt::MakeTensorPtrOp>(
        b.clone(*loadInfo.getBlockPtr().getOperation()));

    Type newType = annotatePrefetchType(ptr.getType(), numWarps);
    ptr.getResult().setType(cast<tt::PointerType>(newType));
    Location loc = ptr.getLoc();

    Value currPtr = ptr;
    for (int i = 0; i < numAdvancePrefetches; ++i) {
      b.create<ttgi::PrefetchOp>(loc, currPtr, load.getCache(), load.getEvict(),
                                 load.getIsVolatile());
      currPtr = b.create<tt::AdvanceOp>(loc, currPtr.getType(), currPtr,
                                        loadInfo.getOffsets());
    }

    prefetchPtrs.push_back(currPtr);
  }
}

void PrefetchBlockPass::injectPrefetchOpsInBody(
    scf::ForOp loop, SmallVectorImpl<Value> &prefetchPtrs) const {
  assert(!prefetchPtrs.empty() && "Expecting an non-empty vector");

  OpBuilder b(loop);
  SmallVector<Value> iterArgs = loop.getInitArgs();
  const size_t num = iterArgs.size();
  iterArgs.append(prefetchPtrs);

  auto newLoop =
      b.create<scf::ForOp>(loop.getLoc(), loop.getLowerBound(),
                           loop.getUpperBound(), loop.getStep(), iterArgs);
  auto args = newLoop.getBody()->getArguments();

  for (auto [lhs, rhs] :
       llvm::zip(loop.getBody()->getArguments(), args.take_front(num + 1)))
    lhs.replaceAllUsesWith(rhs);
  loop.replaceAllUsesWith(newLoop.getResults().take_front(num));
  newLoop.getBody()->getOperations().splice(std::prev(newLoop.getBody()->end()),
                                            loop.getBody()->getOperations());
  auto yield = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
  loop.erase();

  SmallVector<Value> advances;
  unsigned i = 0;
  for (tt::LoadOp load : loopLoads.at(loop)) {
    const LoadInfo &loadInfo = loadToLoadInfo.at(load);
    // FIXME: add a named barrier to increase performance
    // if (i == 0)
    //   b.create<gpu::BarrierOp>(loc);

    b.setInsertionPoint(loadInfo.getAdvance());
    Location loc = loadInfo.getAdvance().getLoc();

    b.create<ttgi::PrefetchOp>(loc, args[num + 1 + i], load.getCache(),
                               load.getEvict(), load.getIsVolatile());
    auto advance =
        b.create<tt::AdvanceOp>(loc, args[num + 1 + i].getType(),
                                args[num + 1 + i], loadInfo.getOffsets());
    advances.push_back(advance);
    i++;
  }

  yield.getResultsMutable().append(advances);
}

} // namespace

std::unique_ptr<mlir::Pass>
mlir::triton::gpu::intel::createPrefetchBlockPass() {
  return std::make_unique<PrefetchBlockPass>();
}
