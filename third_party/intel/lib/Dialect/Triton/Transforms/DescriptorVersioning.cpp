//===- DescriptorVersioning.cpp - Version descriptor loops ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Runtime row-length versioning for persistent tensor-descriptor loops.
//
// Clone a persistent scf.while under `gk % d == 0` so the then-branch can use a
// row length rewritten to `(gk / d) * d` (== gk under the predicate, and
// structurally divisible so the 2D-block gate in MaterializeBlockPointer
// fires), while the else-branch keeps the runtime `gk` for the gather path.
// `d` is the surface-width granularity (2 for fp16).
//
//===----------------------------------------------------------------------===//

#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Utils/Utility.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "triton-intel-descriptor-versioning"

using namespace mlir;
namespace tt = mlir::triton;

namespace mlir::triton::intel {
#define GEN_PASS_DEF_TRITONINTELDESCRIPTORVERSIONING
#include "intel/include/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton::intel

namespace {

// Marks a versioned (then/else) clone so it isn't versioned again.
constexpr StringLiteral kVersionedAttr = "ttig.descriptor_k_versioned";

struct DescCandidate {
  tt::MakeTensorDescOp descOp;
  Value innerShape;
  unsigned divisor;
};

SmallVector<DescCandidate> collectKCandidates(scf::WhileOp whileOp) {
  llvm::SmallMapVector<Operation *, DescCandidate, 4> candidates;
  auto consider = [&](Value desc) {
    std::optional<tt::MakeTensorDescOp> descOp =
        tt::intel::findMakeTensorDescOp(desc);
    if (!descOp || candidates.contains(descOp->getOperation()))
      return;
    // The descriptor must dominate the while for the rewritten clone to be
    // valid; in grouped-GEMM it sits just before the while in the same block.
    if (descOp->getOperation()->getBlock() != whileOp->getBlock() ||
        !descOp->getOperation()->isBeforeInBlock(whileOp))
      return;

    Operation::operand_range shape = descOp->getShape();
    unsigned rank = shape.size();
    if (rank < 2)
      return;

    auto descType = cast<tt::TensorDescType>(descOp->getType());
    unsigned divisor =
        llvm::divideCeil(32u, descType.getBlockType().getElementTypeBitWidth());
    if (divisor <= 1)
      return;

    // Only version a runtime row length; a constant already passes the gate.
    Value innerShape = shape[rank - 1];
    if (tt::intel::getFoldedConstantValue(tt::intel::getFinalValue(innerShape)))
      return;

    candidates.insert({descOp->getOperation(), {*descOp, innerShape, divisor}});
  };

  whileOp.walk([&](Operation *op) {
    if (auto load = dyn_cast<tt::DescriptorLoadOp>(op))
      consider(load.getDesc());
    else if (auto store = dyn_cast<tt::DescriptorStoreOp>(op))
      consider(store.getDesc());
  });

  return llvm::to_vector(llvm::make_second_range(candidates));
}

void versionWhileForK(scf::WhileOp whileOp,
                      ArrayRef<DescCandidate> candidates) {
  Location loc = whileOp.getLoc();
  OpBuilder builder(whileOp);

  // One predicate term per distinct row length.
  llvm::MapVector<Value, unsigned> shapeToDivisor;
  for (const DescCandidate &c : candidates) {
    auto [it, inserted] = shapeToDivisor.try_emplace(c.innerShape, c.divisor);
    if (!inserted)
      it->second = std::max(it->second, c.divisor);
  }

  Value zero =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
  Value verCond;
  for (auto &[innerShape, divisor] : shapeToDivisor) {
    Value div = arith::ConstantOp::create(builder, loc,
                                          builder.getI32IntegerAttr(divisor));
    Value rem = arith::RemSIOp::create(builder, loc, innerShape, div);
    Value cmp = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                      rem, zero);
    verCond =
        verCond ? arith::AndIOp::create(builder, loc, verCond, cmp).getResult()
                : cmp;
  }
  assert(verCond && "Expecting at least one versioning condition");

  auto ifOp = scf::IfOp::create(builder, loc, whileOp.getResultTypes(), verCond,
                                /*withElseRegion=*/true);
  OpBuilder thenB = ifOp.getThenBodyBuilder();
  Operation *thenWhile = thenB.clone(*whileOp.getOperation());
  OpBuilder elseB = ifOp.getElseBodyBuilder();
  Operation *elseWhile = elseB.clone(*whileOp.getOperation());
  if (!thenWhile->getResults().empty()) {
    scf::YieldOp::create(thenB, loc, thenWhile->getResults());
    scf::YieldOp::create(elseB, loc, elseWhile->getResults());
  }

  thenWhile->setAttr(kVersionedAttr, builder.getUnitAttr());
  elseWhile->setAttr(kVersionedAttr, builder.getUnitAttr());

  for (auto [orig, repl] : llvm::zip(whileOp.getResults(), ifOp.getResults()))
    orig.replaceAllUsesWith(repl);

  for (const DescCandidate &c : candidates) {
    tt::MakeTensorDescOp descOp = c.descOp;
    OpBuilder descB(descOp);
    descB.setInsertionPointAfter(descOp);
    Value div = arith::ConstantOp::create(descB, loc,
                                          descB.getI32IntegerAttr(c.divisor));
    Value divided = arith::DivSIOp::create(descB, loc, c.innerShape, div);
    Value aligned = arith::MulIOp::create(descB, loc, divided, div);

    auto alignedDesc = cast<tt::MakeTensorDescOp>(descOp->clone());
    descB.insert(alignedDesc);
    MutableOperandRange shapeMut = alignedDesc.getShapeMutable();
    unsigned lastIdx = alignedDesc.getShape().size() - 1;
    unsigned idx = 0;
    for (OpOperand &shapeOperand : shapeMut) {
      if (idx++ == lastIdx) {
        shapeOperand.set(aligned);
        break;
      }
    }

    // Rewire only the then-clone to the aligned descriptor.
    descOp.getResult().replaceUsesWithIf(
        alignedDesc.getResult(), [&](OpOperand &use) {
          return thenWhile->isProperAncestor(use.getOwner());
        });
  }

  whileOp.erase();
}

struct TritonIntelDescriptorVersioning
    : tt::intel::impl::TritonIntelDescriptorVersioningBase<
          TritonIntelDescriptorVersioning> {
public:
  using Base::Base;

  void runOnOperation() final {
    ModuleOp moduleOp = getOperation();

    // Collect first, then transform: versioning erases the whole nest, so only
    // queue outermost, unversioned loops.
    SmallVector<scf::WhileOp> whileWorklist;
    moduleOp.walk([&](scf::WhileOp whileOp) {
      if (!whileOp->hasAttr(kVersionedAttr) &&
          !whileOp->getParentOfType<scf::WhileOp>())
        whileWorklist.push_back(whileOp);
    });
    for (scf::WhileOp whileOp : whileWorklist) {
      SmallVector<DescCandidate> candidates = collectKCandidates(whileOp);
      if (candidates.empty())
        continue;
      LLVM_DEBUG(llvm::dbgs() << "K-versioning scf.while with "
                              << candidates.size() << " candidate(s)\n");
      versionWhileForK(whileOp, candidates);
    }

    LLVM_DEBUG(llvm::dbgs() << "After versioning:\n" << moduleOp << "\n");
    assert(succeeded(verify(moduleOp)) && "Module verification failed");
  }
};

} // namespace
