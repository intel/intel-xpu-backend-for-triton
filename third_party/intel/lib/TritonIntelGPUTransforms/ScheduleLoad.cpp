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
        dots.push_back(dot);
        group = currGroup;
      }
      assert(!dots.empty() && "No dot found in the loop");
      dotsGroup.push_back(dots);

      for (SmallVector<tt::DotOp> &dots : dotsGroup) {
        SmallVector<Value> notVisited = getNotVisitedUses(dots, 0);
        notVisited.append(getNotVisitedUses(dots, 1));
        for (Value val : notVisited) {
          Operation *op = val.getDefiningOp();
          op->moveBefore(dots.begin()->getOperation());
        }
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
  // Backtrace to collect unvisited dot operands
  // Only handle dot operands which is from tt.load and ttgi.extract
  void markUnvisited(Value val, SmallVector<Value> &notVisited) {
    if (visited.contains(val))
      return;
    if (auto load = dyn_cast<tt::LoadOp>(val.getDefiningOp())) {
      notVisited.push_back(val);
    } else if (auto extract = dyn_cast<ttgi::ExtractOp>(val.getDefiningOp())) {
      Value base = extract.getBase();
      markUnvisited(base, notVisited);
      notVisited.push_back(val);
    }
    visited.insert(val);
  }

  // hack!!! only trace dot A/B, only back 1 level
  SmallVector<Value> getNotVisitedUses(SmallVector<tt::DotOp> &dots,
                                       unsigned opIdx) {
    assert((opIdx == 1 || opIdx == 0) && "opIdx should be 0 or 1");

    SmallVector<Value> notVisited;
    for (tt::DotOp &dot : dots) {
      Value val = (opIdx == 1) ? dot.getB() : dot.getA();
      markUnvisited(val, notVisited);
    }
    return notVisited;
  }

  DenseSet<Value> visited;
};

} // namespace
