//===- ScheduleLoad.cpp ----------------------------------------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements a naive scheduler for loop with load/dot to help IGC's
/// Register Allocation.
/// For now, we put loads adjacent to its user dot.
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"

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
namespace ttgi = mlir::triton::gpu::intel;

#define DEBUG_TYPE "tritonintelgpu-schedule-load"

namespace {

class ScheduleLoadPass
    : public triton::gpu::intel::impl::TritonIntelGPUScheduleLoadBase<
          ScheduleLoadPass> {
public:
  void runOnOperation() override {
    if (!triton::tools::getBoolEnv("TRITON_INTEL_ENABLE_INSTR_SCHED"))
      return;

    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    mod.walk<WalkOrder::PreOrder>([&](scf::ForOp loop) {
      visited.clear();
      int group = -1;
      unsigned numGroups = 0;
      SmallVector<SmallVector<tt::DotOp>> dotsGroup;
      SmallVector<tt::DotOp> dots;
      for (auto dot : loop.getOps<tt::DotOp>()) {
        auto groupAttr = dot->getAttrOfType<IntegerAttr>("schedule-group");
        int currGroup = groupAttr.getInt();
        // a new set of schedule-group start (e.g. 0000 - 1111)
        if (currGroup != group && !dots.empty()) {
          dotsGroup.push_back(dots);
          dots.clear();
        }
        // mark first dot B as visited to not move
        if (currGroup == 0) {
          SmallVector<tt::DotOp> vec{dot};
          getNotVisitedUses(vec, 1);
          // a new set of schedule-groups(e.g. 0000,1111,2222,3333 - 0000) start
          if (group > 0)
            numGroups = dotsGroup.size();
        }
        dots.push_back(dot);
        group = currGroup;
      }
      assert(!dots.empty() && "No dot found in the loop");
      dotsGroup.push_back(dots);

      unsigned i = 0;
      Operation *start = &loop.getBody()->front();
      for (SmallVector<tt::DotOp> &dots : dotsGroup) {
        auto notVisited = getNotVisitedUses(dots, 1);
        if (i == 0)
          notVisited.append(getNotVisitedUses(dots, 0));
        for (Value val : notVisited) {
          auto op = val.getDefiningOp();
          if (i == 0)
            op->moveBefore(start);
          else
            op->moveBefore(dots.begin()->getOperation());
        }
        i++;
        if (i == numGroups)
          i = 0;
      }
    });

    // HoHo, move trunc forward
    mod.walk([&](arith::TruncFOp op) {
      auto def = op.getIn().getDefiningOp();
      op->moveAfter(def);
    });

    // HoHo, add fastmath for all
    // may do this after llvm ir according to user fmath flag
    mod.walk([&](Operation *op) {
      if (auto fmIf = dyn_cast<arith::ArithFastMathInterface>(op))
        op->setAttr(
            fmIf.getFastMathAttrName(),
            arith::FastMathFlagsAttr::get(ctx, arith::FastMathFlags::fast));
    });
  }

private:
  // hack!!! only trace dot A/B, only back 1 level
  SmallVector<Value> getNotVisitedUses(SmallVector<tt::DotOp> &dots,
                                       unsigned opIdx) {
    assert((opIdx == 1 || opIdx == 0) && "opIdx should be 0 or 1");

    SmallVector<Value> notVisited;
    for (tt::DotOp &dot : dots) {
      Value val = (opIdx == 1) ? dot.getB() : dot.getA();
      if (visited.contains(val))
        continue;

      Operation *def = val.getDefiningOp();
      if (auto extract = dyn_cast<ttgi::ExtractOp>(def)) {
        Value base = extract.getBase();
        if (!visited.contains(base)) {
          notVisited.push_back(base);
          visited.insert(base);
        }
      }
      notVisited.push_back(val);
      visited.insert(val);
    }
    return notVisited;
  }

  DenseSet<Value> visited;
};

} // namespace
