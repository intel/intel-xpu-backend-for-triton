//===-------------- PrefetchBlock.cpp -  ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a pass to add prefetch mechanism for target that
/// supports memory prefetch.
/// This pass match certain pattern of scf.loop with tt.load and then add
/// prefetch both in the loop preheader(3 stages in advance) and loop body.
/// This pass only support cases with block pointer.
/// This pass should be run after triton-to-tritongpu-warp.
/// This pass also add layout Attr to newly created ops.
///
/// before this pass:
///  scf.for
///    tt.load
///    tt.advance
///
/// after this pass:
/// tt.make_tensor_ptr
/// tt.prefetch
/// tt.advance
///  scf.for
///    tt.load
///    tt.advance
///    tt.prefetch
///    tt.advance
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include <memory>

namespace mlir {
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttig = mlir::triton::gpu::intel;

namespace {
struct LoadInfo {
  tt::LoadOp load;
  tt::AdvanceOp advance;
  SmallVector<Value> offsets;
  tt::MakeTensorPtrOp blockPtr;
};
void expandDefChain(scf::ForOp loop, Value val, tt::MakeTensorPtrOp &blockPtr) {
  Dialect *arithDialect = val.getContext()->getLoadedDialect("arith");
  Dialect *mathDialect = val.getContext()->getLoadedDialect("math");
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
    expandDefChain(loop, loopArg, blockPtr);
  } else if (auto op = val.getDefiningOp()) {
    if (auto makePtrOp = dyn_cast<tt::MakeTensorPtrOp>(op)) {
      blockPtr = makePtrOp;
    } else if (auto Advance = dyn_cast<tt::AdvanceOp>(op)) {
      // handle it later;
    } else if (op->getDialect() == arithDialect ||
               op->getDialect() == mathDialect) {
      // handle it later;
    }
  }
  return;
}

Type annotatePrefetchType(Type type, unsigned numWarps) {
  RankedTensorType tType;
  auto ptrType = dyn_cast<tt::PointerType>(type);
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    tType = tensorType;
  } else if (ptrType) {
    tType = cast<RankedTensorType>(ptrType.getPointeeType());
  }
  auto shape = tType.getShape();
  assert(shape.size() == 2 && "add more support other than 2d shape");
  SmallVector<unsigned> sizePerWarp(2);
  SmallVector<unsigned> warpsPerCTA(2);
  auto m = shape[0];
  auto n = shape[1];

  // typical numWarps 4, 8, 16, 32, 64
  // naive way to get warp distribute
  auto sizeX = n < 32 ? n : 32; // elementtype
  auto numWarpsX = n / sizeX;
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
  else
    return newType;
}

class PrefetchBlockPass
    : public TritonIntelGPUPrefetchBlockBase<PrefetchBlockPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ModuleOp mod = getOperation();

    for (auto func : mod.getOps<tt::FuncOp>()) {
      bool hasBlockLoadInLoop = false;
      DenseMap<scf::ForOp, SmallVector<tt::LoadOp>> loopLoads;
      // collect info
      auto result = func.walk([&](Operation *op) -> WalkResult {
        if (auto load = dyn_cast<tt::LoadOp>(op)) {
          if (!isa<tt::PointerType>(load.getPtr().getType()))
            return WalkResult::interrupt();
          // assume loop is immediate parent of load for now
          if (auto loop = dyn_cast<scf::ForOp>(load->getParentOp())) {
            hasBlockLoadInLoop = true;
            loopLoads[loop].push_back(load);
          }
        }
        return WalkResult::advance();
      });
      if (result == WalkResult::interrupt() || !hasBlockLoadInLoop)
        return;

      // match load pattern
      // scf.for ...      iter_args(%ptr = %init)
      //   %ld = tt.load %ptr
      //   ...
      //   %newPtr = tt.advance %ptr
      for (auto [loop, loads] : loopLoads) {
        SmallVector<LoadInfo> loadInfos;
        for (auto load : loads) {
          LoadInfo loadInfo;
          loadInfo.load = load;
          for (auto user : load.getPtr().getUsers()) {
            if (user == load)
              continue;
            else if (auto advance = dyn_cast<tt::AdvanceOp>(user))
              loadInfo.advance = advance;
            else
              assert(0 && "not considered case");
          }
          if (!loadInfo.advance)
            continue;
          SmallVector<OpFoldResult> rawOffsets = loadInfo.advance.getOffsets();
          auto offsets = getConstantIntValues(rawOffsets);
          if (!offsets)
            continue;
          llvm::transform(rawOffsets, std::back_inserter(loadInfo.offsets),
                          [&](OpFoldResult ofr) { return cast<Value>(ofr); });
          expandDefChain(loop, load.getPtr(), loadInfo.blockPtr);
          if (!loadInfo.blockPtr)
            continue;
          loadInfos.push_back(loadInfo);
        }

        /// add prefetch in the loop pre-header
        OpBuilder b(loop);
        auto loc = loop.getLoc();
        SmallVector<Value> prefetchPtrs;
        for (auto loadInfo : loadInfos) {
          b.setInsertionPoint(loadInfo.blockPtr);
          auto clone = b.clone(*loadInfo.blockPtr.getOperation());
          auto ptr = cast<tt::MakeTensorPtrOp>(clone);
          auto numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
          auto newType = annotatePrefetchType(ptr.getType(), numWarps);
          ptr.getResult().setType(cast<tt::PointerType>(newType));
          loc = ptr.getLoc();
          // prefetch 3 stages in advance
          auto load = loadInfo.load;
          auto prefetch0 = b.create<ttig::PrefetchOp>(
              loc, ptr, load.getCache(), load.getEvict(), load.getIsVolatile());
          auto prePtr0 = b.create<tt::AdvanceOp>(loc, ptr.getType(), ptr,
                                                 loadInfo.offsets);
          auto prefetch1 =
              b.create<ttig::PrefetchOp>(loc, prePtr0, load.getCache(),
                                         load.getEvict(), load.getIsVolatile());
          auto prePtr1 = b.create<tt::AdvanceOp>(loc, ptr.getType(), prePtr0,
                                                 loadInfo.offsets);
          auto prefetch2 =
              b.create<ttig::PrefetchOp>(loc, prePtr1, load.getCache(),
                                         load.getEvict(), load.getIsVolatile());
          auto prePtr2 = b.create<tt::AdvanceOp>(loc, ptr.getType(), prePtr1,
                                                 loadInfo.offsets);
          prefetchPtrs.push_back(prePtr2);
        }

        /// mutate loop
        b.setInsertionPoint(loop);
        loc = loop.getLoc();
        SmallVector<Value> iterArgs = loop.getInitArgs();
        auto num = iterArgs.size();
        iterArgs.append(prefetchPtrs);
        auto newLoop = b.create<scf::ForOp>(loc, loop.getLowerBound(),
                                            loop.getUpperBound(),
                                            loop.getStep(), iterArgs);
        auto args = newLoop.getBody()->getArguments();
        for (auto [lhs, rhs] : llvm::zip(loop.getBody()->getArguments(),
                                         args.take_front(num + 1)))
          lhs.replaceAllUsesWith(rhs);
        loop.replaceAllUsesWith(newLoop.getResults().take_front(num));
        newLoop.getBody()->getOperations().splice(
            std::prev(newLoop.getBody()->end()),
            loop.getBody()->getOperations());
        auto yield = cast<scf::YieldOp>(newLoop.getBody()->getTerminator());
        loop.erase();

        /// add prefetch in the loop body
        SmallVector<Value> advances;
        for (auto i = 0; i < loadInfos.size(); i++) {
          auto info = loadInfos[i];
          auto load = info.load;
          b.setInsertionPoint(load);
          loc = load.getLoc();
          // FIXME: add barrier every 8 iteration
          // if (i == 0)
          //   b.create<gpu::BarrierOp>(loc);
          b.setInsertionPoint(info.advance);
          loc = info.advance.getLoc();
          auto prefetchInLoop = b.create<ttig::PrefetchOp>(
              loc, args[num + 1 + i], load.getCache(), load.getEvict(),
              load.getIsVolatile());
          auto advance =
              b.create<tt::AdvanceOp>(loc, args[num + 1 + i].getType(),
                                      args[num + 1 + i], info.offsets);
          advances.push_back(advance);
        }
        yield.getResultsMutable().append(advances);
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass>
mlir::triton::gpu::intel::createPrefetchBlockPass() {
  return std::make_unique<PrefetchBlockPass>();
}
