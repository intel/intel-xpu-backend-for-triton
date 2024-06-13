//===- ScheduleLoad.cpp ----------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a naive scheduler for loop with load/dot
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
#define GEN_PASS_DEF_TRITONINTELGPUSCHEDULELOAD
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-schedule-load"

namespace {

class ScheduleLoadPass
    : public triton::gpu::intel::impl::TritonIntelGPUScheduleLoadBase<
          ScheduleLoadPass> {
public:
  SmallVector<Value> getNotVisitedUsesA(SmallVector<tt::DotOp> dots) {
    SmallVector<Value> notVisited;
    for (auto &dot : dots) {
      auto val = dot.getA();
      if (visited.count(val) != 0)
        continue;
      auto def = val.getDefiningOp();
      if (auto extract = dyn_cast<ttgi::ExtractOp>(def)) {
        auto base = extract.getBase();
        if (visited.count(base) == 0) {
          notVisited.push_back(base);
          visited.insert(base);
        }
      }
      notVisited.push_back(val);
      visited.insert(val);
    }
    return notVisited;
  }

  // hack!!! only trace dotB, only back 1 level
  SmallVector<Value> getNotVisitedUses(SmallVector<tt::DotOp> dots) {
    SmallVector<Value> notVisited;
    for (auto &dot : dots) {
      // for (auto operand : dot->getOperands())
      auto val = dot.getB();
      if (visited.count(val) != 0)
        continue;
      auto def = val.getDefiningOp();
      if (auto extract = dyn_cast<ttgi::ExtractOp>(def)) {
        auto base = extract.getBase();
        if (visited.count(base) == 0) {
          notVisited.push_back(base);
          visited.insert(base);
        }
      }
      notVisited.push_back(val);
      visited.insert(val);
    }
    return notVisited;
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    ModuleOp m = getOperation();
    bool enableSLM = mlir::triton::tools::getBoolEnv("ENABLE_SLM");
    if (!enableSLM) {
      m.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
        visited.clear();
        unsigned group = -1;
        SmallVector<SmallVector<tt::DotOp>> dotsGroup;
        SmallVector<tt::DotOp> dots;
        for (auto dot : loop.getOps<tt::DotOp>()) {
          auto groupAttr = dot->getAttrOfType<IntegerAttr>("schedule-group");
          unsigned currGroup = groupAttr.getInt();
          if (currGroup != group && !dots.empty()) {
            dotsGroup.push_back(dots);
            dots.clear();
          }
          if (currGroup == 0)
            getNotVisitedUses({dot});
          dots.push_back(dot);
          group = currGroup;
        }
        assert(!dots.empty());
        dotsGroup.push_back(dots);

        unsigned i = 0;
        Operation *start = &loop.getBody()->front();
        for (auto dots : dotsGroup) {
          auto notVisited = getNotVisitedUses(dots);
          if (i == 0)
            notVisited.append(getNotVisitedUsesA(dots));
          for (auto val : notVisited) {
            auto op = val.getDefiningOp();
            if (i == 0)
              op->moveBefore(start);
            else
              op->moveBefore(dots.begin()->getOperation());
          }
          i++;
          if (i == 4)
            i = 0;
        }
      });
    }

    //
    m.walk([&](arith::TruncFOp op) {
      auto def = op.getIn().getDefiningOp();
      op->moveAfter(def);
    });

    // HoHo, add fastmath for all
    // maybe after llvm ir
    m.walk([&](Operation *op) {
      if (auto fmIf = dyn_cast<arith::ArithFastMathInterface>(op))
        op->setAttr(
            fmIf.getFastMathAttrName(),
            arith::FastMathFlagsAttr::get(ctx, arith::FastMathFlags::fast));
    });
  }

private:
  DenseSet<Value> visited;
};

} // namespace
