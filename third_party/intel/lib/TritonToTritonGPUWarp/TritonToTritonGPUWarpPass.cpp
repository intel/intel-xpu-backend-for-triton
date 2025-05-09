//===- TritonToTritonGPUWarpPass.cpp -  ------------------------*- C++ -*-===//
//
// Copyright 2024 Intel Corporation
// Part of the intel-xpu-backend-for-trito Project, under the Apache License
// v2.0 with LLVM Exceptions. See https://llvm.org/LICENSE.txt for license
// information. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a pass to convert triton to tritongpu with warp
/// distribute annotation. This pass first analyze the kernel's workload
/// pattern (elementwise/reduction/gemm/attention),
/// and then figure out the best layout for key/anchor operation (dot in
/// gemm case). Afterwards, we get all other operation’s layout
/// through def/use chain. Finally, each tensor operation is annotated
/// with layout attribute describing what each warp should do.
//===----------------------------------------------------------------------===//

#include "intel/include/TritonToTritonGPUWarp/TritonToTritonGPUWarpPass.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::intel {
#define GEN_PASS_DECL_CONVERTTRITONTOTRITONGPUWARP
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPUWARP
#include "intel/include/TritonToTritonGPUWarp/Passes.h.inc"
} // namespace mlir::triton::intel

#define DEBUG_TYPE "convert-triton-to-tritongpu-warp"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

// pass named attrs (e.g., tt.contiguity) from Triton to TritonGPU
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

struct DotInfo {
  tt::DotOp dot;
  SmallVector<Value> chainOpsA;
  tt::LoadOp loadA;
  tt::AdvanceOp advanceA;
  SmallVector<Value> chainOpsB;
  tt::LoadOp loadB;
  tt::AdvanceOp advanceB;
  SmallVector<Value> chainOpsC;
  void dump() {
    LLVM_DEBUG(dot.dump());
    LDBG("***** chain ops of dotA *****\n");
    for (auto val : chainOpsA)
      LLVM_DEBUG(val.dump());
    LDBG("***** chain ops end *********\n");
    if (loadA)
      LLVM_DEBUG(loadA.dump());
    if (advanceA)
      LLVM_DEBUG(advanceA.dump());
    LDBG("\n");
    LDBG("***** chain ops of dotB *****\n");
    for (auto val : chainOpsB)
      LLVM_DEBUG(val.dump());
    LDBG("***** chain ops end *********\n");
    if (loadB)
      LLVM_DEBUG(loadB.dump());
    if (advanceB)
      LLVM_DEBUG(advanceB.dump());
    LDBG("\n");
    LDBG("***** chain ops of dotC *****\n");
    for (auto val : chainOpsC)
      LLVM_DEBUG(val.dump());
    LDBG("***** chain ops end *********\n");
  }
};
// only support at most 2 dot in a loop for now
struct LoopDotInfo {
  DotInfo dotInfo0;
  DotInfo dotInfo1;
  bool connectDotA = false;
  bool connectDotB = false;
  bool connectDotC = false;
  void dump() {
    LDBG("\n");
    LDBG("***** first dot info *****\n");
    LLVM_DEBUG(dotInfo0.dump());
    if (dotInfo1.dot) {
      LDBG("\n");
      LDBG("connect to first DotA " << connectDotA << "\n");
      LDBG("connect to first DotB " << connectDotB << "\n");
      LDBG("connect to first DotC " << connectDotC << "\n");
      LDBG("***** second dot info *****\n");
      LLVM_DEBUG(dotInfo1.dump());
    }
  }
};

} // namespace

namespace mlir::triton::intel {
class ConvertTritonToTritonGPUWarp
    : public impl::ConvertTritonToTritonGPUWarpBase<
          ConvertTritonToTritonGPUWarp> {
public:
  using impl::ConvertTritonToTritonGPUWarpBase<
      ConvertTritonToTritonGPUWarp>::ConvertTritonToTritonGPUWarpBase;
  ConvertTritonToTritonGPUWarp() = default;
  ConvertTritonToTritonGPUWarp(unsigned numWarps) { this->numWarps = numWarps; }

private:
  DenseMap<Value, Attribute> valueAttrMap;
  Dialect *arithDialect = nullptr;
  Dialect *mathDialect = nullptr;

public:
  LogicalResult initialize(MLIRContext *context) override {
    arithDialect = context->getLoadedDialect("arith");
    mathDialect = context->getLoadedDialect("math");
    valueAttrMap.clear();
    return success();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();
    auto i32Ty = IntegerType::get(ctx, 32);
    // sub-function should be inlined up to now
    for (auto func : mod.getOps<tt::FuncOp>()) {
      bool hasBlockPointer = false;
      bool hasDot = false;
      SmallVector<scf::ForOp> loops;
      // get info of block pointer and dot
      auto result = func.walk([&](Operation *op) -> WalkResult {
        if (auto load = dyn_cast<tt::LoadOp>(op)) {
          if (!isa<tt::PointerType>(load.getPtr().getType()))
            return WalkResult::interrupt();
        } else if (isa<tt::MakeTensorPtrOp>(op))
          hasBlockPointer = true;
        else if (isa<tt::DotOp>(op))
          hasDot = true;
        else if (auto loop = dyn_cast<scf::ForOp>(op))
          loops.push_back(loop);
        return WalkResult::advance();
      });
      if (result == WalkResult::interrupt() || !hasBlockPointer)
        return;

      /// work-load analysis
      // 4 catagories: elementwise, reduction, gemm, flashattention
      // only handle gemm/flashattention for now
      if (!hasDot || loops.size() == 0)
        return;
      valueAttrMap.clear();
      DenseMap<scf::ForOp, LoopDotInfo> loopMap;
      SmallVector<Workload, 2> workloads;
      for (auto loop : loops) {
        auto dots = llvm::to_vector(loop.getOps<tt::DotOp>());
        assert(dots.size() <= 2 && "only support 1 or 2 dot in a loop");
        LoopDotInfo loopDotInfo;
        collectLoopDotInfo(loop, dots[0], loopDotInfo);
        LLVM_DEBUG(loopDotInfo.dump());
        loopMap[loop] = loopDotInfo;
        // DAG pattern match
        Workload workLoadKind = matchLoopWorkload(loop, loopDotInfo);
        workloads.push_back(workLoadKind);

        /// get tensor layout attr according to workload pattern
        switch (workLoadKind) {
        case Workload::Gemm: {
          auto &info0 = loopDotInfo.dotInfo0;
          auto dot = info0.dot;
          auto aType = cast<RankedTensorType>(dot.getA().getType());
          auto bType = cast<RankedTensorType>(dot.getB().getType());
          auto m = aType.getShape()[0];
          auto n = bType.getShape()[1];
          auto [sizePerWarp, warpsPerCTA] = determineDotConfig(m, n, numWarps);
          auto ctaLayout = ttg::CTALayoutAttr::get(ctx, {1, 1}, {1, 1}, {1, 0});
          auto dotCLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarp, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto dotALayout = ttg::DotOperandEncodingAttr::get(
              ctx, 0, dotCLayout, aType.getElementType());
          auto dotBLayout = ttg::DotOperandEncodingAttr::get(
              ctx, 1, dotCLayout, bType.getElementType());
          // record value's attr
          for (auto op : info0.chainOpsA)
            valueAttrMap[op] = dotALayout;
          for (auto op : info0.chainOpsB)
            valueAttrMap[op] = dotBLayout;
          for (auto op : info0.chainOpsC)
            valueAttrMap[op] = dotCLayout;
          if (info0.advanceA)
            valueAttrMap[info0.advanceA] = dotALayout;
          if (info0.advanceB)
            valueAttrMap[info0.advanceB] = dotBLayout;
          break;
        }
        case Workload::None:
          LDBG("\n");
          LDBG("***********************************************\n");
          LDBG("this has tt.dot, but workload do not match any \n");
          LDBG("***********************************************\n");
          LDBG("\n");
          return;
        case Workload::ElementWise:
        case Workload::Reduction: {
          break;
        }
        case Workload::Attention: {
          DotInfo &info0 = loopDotInfo.dotInfo0;
          DotInfo &info1 = loopDotInfo.dotInfo1;
          DotOp dot0 = info0.dot;
          auto aType = cast<RankedTensorType>(dot0.getA().getType());
          auto bType = cast<RankedTensorType>(dot0.getB().getType());
          unsigned Br = aType.getShape()[0];
          unsigned d = bType.getShape()[0];
          unsigned Bc = bType.getShape()[1];
          assert(Br % numWarps == 0 && "rows should be multiple of numWarps");
          assert(Bc % numWarps == 0 &&
                 "columns should be multiple of numWarps");
          SmallVector<unsigned> warpsPerCTA{numWarps, 1};
          SmallVector<unsigned> sizePerWarpQ{Br / numWarps, d};
          SmallVector<unsigned> sizePerWarpK{d, Bc};
          SmallVector<unsigned> sizePerWarpQK{Br / numWarps, Bc};
          SmallVector<unsigned> sizePerWarpV{Bc, d};
          SmallVector<unsigned> sizePerWarpO{Br / numWarps, d};
          auto ctaLayout = ttg::CTALayoutAttr::get(ctx, {1, 1}, {1, 1}, {1, 0});
          auto oLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpO, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto vLayout = ttg::DotOperandEncodingAttr::get(
              ctx, 1, oLayout, aType.getElementType());
          auto qkLayout1 = ttg::DotOperandEncodingAttr::get(
              ctx, 0, oLayout, aType.getElementType());
          OpBuilder b(info1.dot);
          auto dot1A = info1.dot.getA();
          auto cvtType = addAttrToType(dot1A.getType(), qkLayout1);
          // add convert layout op for dot1.A
          auto cvt = b.create<ttg::ConvertLayoutOp>(info1.dot.getLoc(), cvtType,
                                                    dot1A);
          dot1A.replaceAllUsesExcept(cvt, cvt);
          auto qkLayout0 = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpQK, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto qLayout = ttg::DotOperandEncodingAttr::get(
              ctx, 0, qkLayout0, aType.getElementType());
          auto kLayout = ttg::DotOperandEncodingAttr::get(
              ctx, 1, qkLayout0, aType.getElementType());

          // record value's attr
          for (auto val : info0.chainOpsA)
            valueAttrMap[val] = qLayout;
          for (auto val : info0.chainOpsB)
            valueAttrMap[val] = kLayout;
          for (auto val : info0.chainOpsC)
            valueAttrMap[val] = qkLayout0;
          if (info0.advanceA)
            valueAttrMap[info0.advanceA] = qLayout;
          if (info0.advanceB)
            valueAttrMap[info0.advanceB] = kLayout;

          assert(info1.chainOpsA.empty());
          for (auto val : info1.chainOpsB)
            valueAttrMap[val] = vLayout;
          {
            Value val = info1.dot;
            DenseSet<Value> chainedVals;
            chainedVals.insert(val);
            expandUseChain(val, chainedVals);
            for (auto val : chainedVals) {
              valueAttrMap[val] = oLayout;
            }
          }
          for (auto val : info1.chainOpsC) {
            if (valueAttrMap.count(val) == 0) {
              valueAttrMap[val] = oLayout;
            } else if (valueAttrMap[val] == oLayout) {
              continue;
            } else {
              // clone value if it has more than 1 layout used
              if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
                OpBuilder b(cst);
                auto newOp = b.clone(*cst);
                auto result = newOp->getResults()[0];
                valueAttrMap[result] = oLayout;
                val.replaceUsesWithIf(result, [&](OpOperand &use) {
                  Operation *user = use.getOwner();
                  auto val = user->getResults()[0];
                  if (std::find(info1.chainOpsC.begin(), info1.chainOpsC.end(),
                                val) != info1.chainOpsC.end())
                    return true;
                  return false;
                });
              } else {
                assert(0 && "add more support");
              }
            }
          }
          assert(!info1.advanceA);
          if (info1.advanceB)
            valueAttrMap[info1.advanceB] = vLayout;
          break;
        }
        }

        loop->setAttr(AttrWorkloadName,
                      IntegerAttr::get(i32Ty, int64_t(workLoadKind)));
      }

      auto opHasTensorType = [&](Operation *op) {
        auto oprndHasTensorType =
            llvm::any_of(op->getOperandTypes(), isTensorOrTensorPointerType);
        auto resultHasTensorType =
            llvm::any_of(op->getResultTypes(), isTensorOrTensorPointerType);
        return oprndHasTensorType || resultHasTensorType;
      };

      /// get other value's layout attr by def/use chain
      func.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (auto loop = dyn_cast<scf::ForOp>(op))
          return;
        else if (!opHasTensorType(op))
          return;
        else if (auto reduce = dyn_cast<tt::ReduceOp>(op)) {
          assert(reduce.getSrcs().size() == 1);
          auto axis = reduce.getAxis();
          auto src = reduce.getSrcs()[0];
          assert(valueAttrMap.count(src) != 0 &&
                 "reduce source attr should be already figured out");
          auto sliceAttr = ttg::SliceEncodingAttr::get(
              ctx, axis,
              cast<mlir::triton::gpu::DistributedEncodingTrait>(
                  valueAttrMap[src]));
          auto result = reduce.getResults()[0];
          DenseSet<Value> chainedVals;
          chainedVals.insert(result);
          expandUseChain(result, chainedVals);
          for (auto val : chainedVals) {
            valueAttrMap[val] = sliceAttr;
          }
        } else if (op->getDialect() == arithDialect ||
                   op->getDialect() == mathDialect) {
          // FIXME: this is really ad-hoc to amend
          if (auto mul = dyn_cast<arith::MulFOp>(op)) {
            auto rhs = mul.getRhs();
            valueAttrMap[rhs] = valueAttrMap[op->getResults()[0]];
          }
        }
      });

      /// adding tensor layout attr to related ops
      func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        if (!opHasTensorType(op))
          return WalkResult::advance();

        unsigned numResults = op->getResults().size();
        if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
          transformArithConstantOp(cst, valueAttrMap[cst]);
        } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
          transformScfForOp(loop);
        } else if (auto store = dyn_cast<tt::StoreOp>(op)) {
          transformStoreOp(store);
        } else if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(op)) {
          ;
          // arith, math, tt::ExpandDimsOp, tt::SplatOp
        } else if (numResults != 0) {
          assert(numResults == 1 && "only support 1 result");
          transformGenericOp(op, valueAttrMap);
        }
        return WalkResult::advance();
      });

      if (loops.size() == 2 && workloads.front() == Workload::Attention &&
          workloads.back() == Workload::Attention) {
        // match attention with causal masking
        // FIXME: This is a workaround to attach layouts to tensor ops that have
        //        not been handled before. This should instead be covered by a
        //        more generic layout propagation approach.
        Attribute blockLayout = loopMap[loops.front()]
                                    .dotInfo0.dot.getResult()
                                    .getType()
                                    .getEncoding();

        func.walk<WalkOrder::PreOrder>([&](Operation *op) {
          SmallVector<RankedTensorType> typesWithoutEncoding;
          for (Type ty : op->getResultTypes()) {
            if (auto tty = dyn_cast<RankedTensorType>(ty))
              if (!tty.getEncoding())
                typesWithoutEncoding.push_back(tty);
          }

          if (typesWithoutEncoding.empty())
            return;

          if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
            transformArithConstantOp(cst, blockLayout);
            return;
          }

          // Assign:
          // - rank-2 operations: block layout
          // - rank-1 operations: slice layout
          assert(op->getNumResults() == 1 &&
                 "Unexpected tensor operation with multiple results");
          OpResult res = op->getOpResult(0);
          auto tty = cast<RankedTensorType>(res.getType());
          if (tty.getRank() == 2)
            res.setType(addAttrToType(tty, blockLayout));
          // Rank==1 tensors get a slice layout with the axis depending on the
          // use.
          if (auto expand = dyn_cast<tt::ExpandDimsOp>(op)) {
            Attribute sliceLayout = triton::gpu::SliceEncodingAttr::get(
                blockLayout.getContext(), expand.getAxis(),
                cast<mlir::triton::gpu::DistributedEncodingTrait>(blockLayout));
            DenseSet<Value> chainedVals;
            expandDefChain(expand.getSrc(), chainedVals);
            for (auto cv : chainedVals)
              cv.setType(addAttrToType(cv.getType(), sliceLayout));
          }
        });
      }
    }

    /// adding module attributes
    mod->setAttr(ttg::AttrNumWarpsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, numWarps.getValue())));
    mod->setAttr(ttg::AttrNumThreadsPerWarp,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
    mod->setAttr(ttg::AttrNumCTAsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
  }

  void transformGenericOp(Operation *op, DenseMap<Value, Attribute> &map) {
    auto result = op->getResults()[0];
    // if already got
    if (map.count(result) != 0) {
      auto newType = addAttrToType(result.getType(), map[result]);
      result.setType(newType);
    }
    // get the attr by propagating
    else if (op->getDialect() == arithDialect ||
             op->getDialect() == mathDialect || isa<tt::BroadcastOp>(op)) {
      Attribute attr;
      for (auto operand : op->getOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
          if (type.getEncoding())
            attr = type.getEncoding();
      }
      auto newType = addAttrToType(result.getType(), attr);
      result.setType(newType);
    } else if (auto expand = dyn_cast<tt::ExpandDimsOp>(op)) {
      auto src = expand.getSrc();
      if (auto attr = dyn_cast_if_present<ttg::SliceEncodingAttr>(
              src.getType().getEncoding())) {
        Type newType = addAttrToType(result.getType(), attr.getParent());
        result.setType(newType);
      }
      // else: will patch the encoding later in the causal-attention-specific
      // layout propagation.
      // FIXME: Remove this workaround.
    }
    // relax upstream broadcast constraint
    if (auto bc = dyn_cast<tt::BroadcastOp>(op)) {
      OpBuilder b(op);
      auto newOp = b.create<ttgi::BroadcastOp>(op->getLoc(), result.getType(),
                                               op->getOperands());
      op->replaceAllUsesWith(newOp);
      op->erase();
    }
  }

  // assume the below code sequence for now
  // %ptr = tt.make_tensor_ptr
  // tt.store %ptr, %value
  void transformStoreOp(tt::StoreOp op) {
    auto attr = cast<RankedTensorType>(op.getValue().getType()).getEncoding();
    if (auto makePtrOp = op.getPtr().getDefiningOp<tt::MakeTensorPtrOp>()) {
      auto result = makePtrOp.getResult();
      auto newType = addAttrToType(result.getType(), attr);
      result.setType(cast<tt::PointerType>(newType));
    }
  }
  void transformScfForOp(scf::ForOp op) {
    auto body = op.getBody();
    for (auto [lhs, rhs] :
         llvm::zip(body->getArguments().drop_front(1), op.getInitArgs()))
      lhs.setType(rhs.getType());
    for (auto i = 0; i < op->getResults().size(); i++) {
      auto init = op.getInitArgs()[i];
      auto type = init.getType();
      op->getResult(i).setType(type);
    }
    return;
  }

  void transformArithConstantOp(arith::ConstantOp op, Attribute attr) {
    auto newType = addAttrToType(op.getType(), attr);
    auto value = cast<DenseElementsAttr>(op.getValue());
    value = value.reshape(cast<ShapedType>(newType));
    OpBuilder b(op);
    auto newOp = b.create<arith::ConstantOp>(op.getLoc(), newType, value);
    addNamedAttrs(newOp, op->getAttrDictionary());
    op->replaceAllUsesWith(newOp->getResults());
    op->erase();
    return;
  }

  Type addAttrToType(Type type, Attribute attr) {
    if (auto tensorType = dyn_cast<RankedTensorType>(type))
      return RankedTensorType::get(tensorType.getShape(),
                                   tensorType.getElementType(), attr);
    else if (auto ptrType = dyn_cast<tt::PointerType>(type)) {
      auto newPointeeType = addAttrToType(ptrType.getPointeeType(), attr);
      return tt::PointerType::get(newPointeeType, ptrType.getAddressSpace());
    }
    return type;
  }

  std::pair<SmallVector<unsigned>, SmallVector<unsigned>>
  determineDotConfig(unsigned m, unsigned n, unsigned numWarps) {
    // typical numWarps 4, 8, 16, 32, 64
    SmallVector<unsigned> sizePerWarp(2);
    SmallVector<unsigned> warpsPerCTA(2);

    // assume llvm dot size is 8x16
    //((n % 16) == 0 && (m % 8) == 0)
    int root = std::sqrt(numWarps);
    int numWarpsX = 1;
    if (n / 16 <= root)
      numWarpsX = n / 16;
    else if (n / 32 <= root)
      numWarpsX = n / 32;
    else if (n / 64 <= root)
      numWarpsX = n / 64;
    else
      numWarpsX = n / 128;
    warpsPerCTA[1] = std::max(numWarpsX, 1);
    warpsPerCTA[0] = ceil<unsigned>(numWarps, warpsPerCTA[1]);
    sizePerWarp[1] = ceil<unsigned>(n, warpsPerCTA[1]);
    sizePerWarp[0] = ceil<unsigned>(m, warpsPerCTA[0]);
    if (sizePerWarp[0] < 8) {
      sizePerWarp[0] = 8;
      warpsPerCTA[0] = 1;
      warpsPerCTA[1] = numWarps;
      sizePerWarp[1] = ceil<unsigned>(n, warpsPerCTA[1]);
    }

    return {sizePerWarp, warpsPerCTA};
  }

  Workload matchLoopWorkload(scf::ForOp loop, LoopDotInfo &loopDotInfo) {
    // match gemm pattern
    //  scf.for idx
    //    %a = tt.load %ptrA
    //    %b = tt.load %ptrB
    //    %c = tt.dot %a, %b, %acc
    //    tt.advance %ptrA, [0, stepA]
    //    tt.advance %ptrB, [stepB, 0]
    if (loopDotInfo.dotInfo0.dot && !loopDotInfo.dotInfo1.dot) {
      auto &dotInfo = loopDotInfo.dotInfo0;
      if (!dotInfo.advanceA || !dotInfo.advanceB)
        return Workload::None;
      SmallVector<OpFoldResult> rawOffsetsA = dotInfo.advanceA.getOffsets();
      SmallVector<OpFoldResult> rawOffsetsB = dotInfo.advanceB.getOffsets();
      auto offsetsA = *getConstantIntValues(rawOffsetsA);
      auto offsetsB = *getConstantIntValues(rawOffsetsB);
      if (offsetsA.size() == 2 && offsetsB.size() == 2 && offsetsA[0] == 0 &&
          offsetsB[1] == 0)
        return Workload::Gemm;
    }
    // match attention qkv pattern
    // %q
    //  scf.for idx
    //    %k = tt.load %ptrK
    //    %s = tt.dot %q, %k
    //    %ss = arit/math  %s
    //    %v = tt.load %ptrV
    //    %o = tt.dot %ss, %v
    //    tt.advance %ptrK, [stepK, 0]
    //    tt.advance %ptrV, [0, stepV]
    else if (loopDotInfo.dotInfo0.dot && loopDotInfo.dotInfo1.dot) {
      if (!loopDotInfo.connectDotA || loopDotInfo.connectDotB)
        return Workload::None;
      auto &info0 = loopDotInfo.dotInfo0;
      auto &info1 = loopDotInfo.dotInfo1;
      if (!info0.chainOpsA.empty()) {
        // Q is loop invariant
        if (Operation *op = info0.chainOpsA[0].getDefiningOp()) {
          if (op->isBeforeInBlock(loop) && info0.advanceB && info1.advanceB) {
            SmallVector<OpFoldResult> rawOffsetsK = info0.advanceB.getOffsets();
            SmallVector<OpFoldResult> rawOffsetsV = info1.advanceB.getOffsets();
            auto offsetsK = *getConstantIntValues(rawOffsetsK);
            auto offsetsV = *getConstantIntValues(rawOffsetsV);
            if (offsetsK.size() == 2 && offsetsV.size() == 2 &&
                offsetsK[0] == 0 && offsetsV[1] == 0 &&
                offsetsK[1] == offsetsV[0])
              return Workload::Attention;
          }
        }
      }
    }
    return Workload::None;
  }

  void collectLoopDotInfo(scf::ForOp loop, tt::DotOp dot,
                          LoopDotInfo &loopDotInfo) {
    auto a = dot.getA();
    auto b = dot.getB();
    auto c = dot.getC();
    auto &info0 = loopDotInfo.dotInfo0;
    info0.dot = dot;
    expandDefChain(loop, a, info0.chainOpsA, info0.loadA, info0.advanceA);
    expandDefChain(loop, b, info0.chainOpsB, info0.loadB, info0.advanceB);
    expandDotCChain(loop, dot, info0.chainOpsC, loopDotInfo);
  }

  void expandDefChain(scf::ForOp loop, Value val, SmallVector<Value> &ops,
                      tt::LoadOp &load, tt::AdvanceOp &advance) {
    ops.push_back(val);
    // // val is loop invariant
    // if (loop.isDefinedOutsideOfLoop(val))
    //   return;
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
      expandDefChain(loop, loopArg, ops, load, advance);
    }
    // defOp inside loop
    else if (auto op = val.getDefiningOp()) {
      if (auto ld = dyn_cast<tt::LoadOp>(op)) {
        // ops.push_back(ld);
        load = ld;
        for (auto user : ld.getPtr().getUsers()) {
          if (user == ld)
            continue;
          else if (auto advanceOp = dyn_cast<tt::AdvanceOp>(user))
            advance = advanceOp;
          else
            assert(0 && "consider more support");
        }
        // block pointer should also be tracked
        expandDefChain(loop, ld.getPtr(), ops, load, advance);
      } else if (auto currAdvance = dyn_cast<tt::AdvanceOp>(op)) {
        expandDefChain(loop, currAdvance.getPtr(), ops, load, advance);
      } else if (op->getDialect() == arithDialect ||
                 op->getDialect() == mathDialect) {
        for (auto operand : op->getOperands()) {
          expandDefChain(loop, operand, ops, load, advance);
        }
      }
    }
    return;
  }

  void expandDotCChain(scf::ForOp loop, tt::DotOp dot, SmallVector<Value> &ops,
                       LoopDotInfo &loopDotInfo) {
    SmallVector<Value> defList;
    tt::LoadOp nullLoad;
    tt::AdvanceOp nullAdv;
    expandDefChain(loop, dot.getC(), defList, nullLoad, nullAdv);
    for (auto op : llvm::reverse(defList))
      ops.push_back(op);
    ops.push_back(dot);
    for (auto it = ++dot->getIterator(); it != loop.end(); it++) {
      auto op = &*it;
      bool inUseChain = llvm::any_of(op->getOperands(), [&](Value val) {
        return std::find(ops.begin(), ops.end(), val) != ops.end();
      });
      if (!inUseChain)
        continue;
      else if (op->getDialect() == arithDialect ||
               op->getDialect() == mathDialect)
        ops.push_back(op->getResults()[0]);
      else if (isa<tt::ReduceOp, tt::ExpandDimsOp, tt::BroadcastOp>(op))
        ;
      else if (auto dot1 = dyn_cast<tt::DotOp>(op)) {
        auto &info1 = loopDotInfo.dotInfo1;
        info1.dot = dot1;
        auto dotA = dot1.getA();
        auto dotB = dot1.getB();
        auto dotC = dot1.getC();
        if (std::find(ops.begin(), ops.end(), dotA) == ops.end())
          expandDefChain(loop, dotA, info1.chainOpsA, info1.loadA,
                         info1.advanceA);
        else
          loopDotInfo.connectDotA = true;
        if (std::find(ops.begin(), ops.end(), dotB) == ops.end())
          expandDefChain(loop, dotB, info1.chainOpsB, info1.loadB,
                         info1.advanceB);
        else
          loopDotInfo.connectDotB = true;
        if (std::find(ops.begin(), ops.end(), dotC) == ops.end())
          expandDotCChain(loop, dot1, info1.chainOpsC, loopDotInfo);
        else
          loopDotInfo.connectDotC = true;
      }
    }
  }

  void expandUseChain(Value val, DenseSet<Value> &chainedVals) {
    for (auto &use : val.getUses()) {
      Operation *op = use.getOwner();
      // arith/math ops
      if (op->getDialect() == arithDialect || op->getDialect() == mathDialect) {
        Value result = op->getResults()[0];
        if (chainedVals.count(result) == 0) {
          chainedVals.insert(result);
          expandUseChain(result, chainedVals);
        }
        for (auto operand : op->getOperands()) {
          expandDefChain(operand, chainedVals);
        }
        // yield
      } else if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        auto loop = cast<scf::ForOp>(yield->getParentOp());
        Value res = loop.getResult(use.getOperandNumber());
        chainedVals.insert(res);
        expandUseChain(res, chainedVals);
        // expanddims, splat, store
      } else if (isa<tt::ExpandDimsOp, tt::SplatOp, tt::StoreOp, scf::ForOp>(
                     op)) {
        continue;
        // other ops
      } else {
        assert(0 && "add more support");
      }
    }
  }

  void expandDefChain(Value val, DenseSet<Value> &chainedVals) {
    if (chainedVals.count(val))
      return;
    chainedVals.insert(val);
    if (auto arg = dyn_cast<BlockArgument>(val)) {
      auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
      assert(loop);
      auto loopArg = loop.getInitArgs()[arg.getArgNumber() - 1];
      expandDefChain(loopArg, chainedVals);
    } else if (auto opRes = dyn_cast<OpResult>(val)) {
      Operation *def = opRes.getOwner();
      if (def->getDialect() == arithDialect ||
          def->getDialect() == mathDialect) {
        for (auto operand : def->getOperands()) {
          expandDefChain(operand, chainedVals);
          expandUseChain(operand, chainedVals);
        }
      } else if (auto forLoop = dyn_cast<scf::ForOp>(def)) {
        Value yieldArg = forLoop.getYieldedValues()[opRes.getResultNumber()];
        chainedVals.insert(yieldArg);
        expandDefChain(yieldArg, chainedVals);
      } else if (isa<tt::SplatOp, tt::BroadcastOp, tt::ReduceOp,
                     tt::MakeRangeOp>(def)) {
        chainedVals.insert(def->getResult(0));
      } else if (isa<tt::ExpandDimsOp>(def)) {
        ;
      } else {
        assert(0 && "add more support");
      }
    } else {
      assert(0 && "add more support");
    }
  }
};

} // namespace mlir::triton::intel
