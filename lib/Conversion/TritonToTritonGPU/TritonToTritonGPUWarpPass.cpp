#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

namespace {
constexpr static char AttrWorkloadName[] = "triton_gpu.workload";

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

enum class Workload {
  // add more
  None = 0, // pattern not match any of below
  ElementWise = 1,
  Reduction = 2,
  Gemm = 3,
  Attention = 4
};

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
    dot.dump();
    llvm::outs() << "***** chain ops of dotA *****\n";
    for (auto val : chainOpsA)
      val.dump();
    llvm::outs() << "***** chain ops end *********\n";
    if (loadA)
      loadA.dump();
    if (advanceA)
      advanceA.dump();
    llvm::outs() << "\n";
    llvm::outs() << "***** chain ops of dotB *****\n";
    for (auto val : chainOpsB)
      val.dump();
    llvm::outs() << "***** chain ops end *********\n";
    if (loadB)
      loadB.dump();
    if (advanceB)
      advanceB.dump();
    llvm::outs() << "\n";
    llvm::outs() << "***** chain ops of dotC *****\n";
    for (auto val : chainOpsC)
      val.dump();
    llvm::outs() << "***** chain ops end *********\n";
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
    llvm::outs() << "\n";
    llvm::outs() << "***** first dot info *****\n";
    dotInfo0.dump();
    if (dotInfo1.dot) {
      llvm::outs() << "\n";
      llvm::outs() << "connect to first DotA " << connectDotA << "\n";
      llvm::outs() << "connect to first DotB " << connectDotB << "\n";
      llvm::outs() << "connect to first DotC " << connectDotC << "\n";
      llvm::outs() << "***** second dot info *****\n";
      dotInfo1.dump();
    }
  }
};

class ConvertTritonToTritonGPUWarp
    : public ConvertTritonToTritonGPUWarpBase<ConvertTritonToTritonGPUWarp> {
public:
  ConvertTritonToTritonGPUWarp() = default;
  ConvertTritonToTritonGPUWarp(int numWarps) { this->numWarps = numWarps; }

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
      DenseMap<Value, Attribute> valueAttrMap;
      DenseMap<scf::ForOp, LoopDotInfo> loopMap;
      for (auto loop : loops) {
        auto dots = llvm::to_vector(loop.getOps<tt::DotOp>());
        assert(dots.size() <= 2 && "only support 1/2 dot in a loop");
        LoopDotInfo loopDotInfo;
        collectLoopDotInfo(loop, dots[0], loopDotInfo);
        loopDotInfo.dump();
        loopMap[loop] = loopDotInfo;
        // DAG pattern match
        auto workLoadKind = matchLoopWorkload(loop, loopDotInfo);
        if (workLoadKind == Workload::None) {
          llvm::outs() << "\n";
          llvm::outs() << "***********************************************\n";
          llvm::outs() << "this has tt.dot, but workload do not match any \n";
          llvm::outs() << "***********************************************\n";
          llvm::outs() << "\n";
          return;
        }
        loop->setAttr(AttrWorkloadName,
                      IntegerAttr::get(i32Ty, int64_t(workLoadKind)));

        /// get tensor layout attr according to workload pattern
        if (workLoadKind == Workload::Gemm) {
          auto &info0 = loopDotInfo.dotInfo0;
          auto dot = info0.dot;
          auto aType = dot.getA().getType().cast<RankedTensorType>();
          auto bType = dot.getB().getType().cast<RankedTensorType>();
          auto m = aType.getShape()[0];
          // auto k = aType.getShape()[1];
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
        } else if (workLoadKind == Workload::Attention) {
          llvm::outs() << "match workload attention \n";
          auto &info0 = loopDotInfo.dotInfo0;
          auto &info1 = loopDotInfo.dotInfo1;
          auto dot0 = info0.dot;
          auto aType = dot0.getA().getType().cast<RankedTensorType>();
          auto bType = dot0.getB().getType().cast<RankedTensorType>();
          unsigned Br = aType.getShape()[0];
          unsigned d = bType.getShape()[0];
          unsigned Bc = bType.getShape()[1];
          assert(Br % numWarps == 0);
          assert(Bc % numWarps == 0);
          SmallVector<unsigned> warpsPerCTA{numWarps, 1};
          SmallVector<unsigned> sizePerWarpQ{Br / numWarps, d};
          SmallVector<unsigned> sizePerWarpK{d, Bc};
          SmallVector<unsigned> sizePerWarpQK{Br / numWarps, Bc};
          SmallVector<unsigned> sizePerWarpV{Bc, d};
          SmallVector<unsigned> sizePerWarpO{Br / numWarps, d};
          auto ctaLayout = ttg::CTALayoutAttr::get(ctx, {1, 1}, {1, 1}, {1, 0});
          auto qLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpQ, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto kLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpK, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto qkLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpK, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto vLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpV, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);
          auto oLayout = ttg::BlockedEncodingAttr::get(
              ctx, sizePerWarpO, {1, 1}, warpsPerCTA, {1, 0}, ctaLayout);

          // record value's attr
          // fixme: the 2nd loop may overwrite, not check it for now
          for (auto op : info0.chainOpsA)
            valueAttrMap[op] = qLayout;
          for (auto op : info0.chainOpsB)
            valueAttrMap[op] = kLayout;
          for (auto op : info0.chainOpsC)
            valueAttrMap[op] = qkLayout;
          if (info0.advanceA)
            valueAttrMap[info0.advanceA] = qLayout;
          if (info0.advanceB)
            valueAttrMap[info0.advanceB] = kLayout;

          assert(info1.chainOpsA.empty());
          for (auto op : info1.chainOpsB)
            valueAttrMap[op] = vLayout;
          for (auto op : info1.chainOpsC)
            valueAttrMap[op] = oLayout;
          assert(!info1.advanceA);
          if (info1.advanceB)
            valueAttrMap[info1.advanceB] = vLayout;
        }
      }

      /// adding tensor layout attr to related ops
      func.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
        auto hasTensorType = [&](Type type) {
          if (isa<RankedTensorType>(type))
            return true;
          else if (auto ptrType = dyn_cast<tt::PointerType>(type))
            if (isa<RankedTensorType>(ptrType.getPointeeType()))
              return true;
          return false;
        };
        auto oprndHasTensorType =
            llvm::any_of(op->getOperandTypes(), hasTensorType);
        auto resultHasTensorType =
            llvm::any_of(op->getResultTypes(), hasTensorType);
        if (!oprndHasTensorType && !resultHasTensorType)
          return WalkResult::advance();

        auto numResults = op->getResults().size();
        if (auto cst = dyn_cast<arith::ConstantOp>(op)) {
          transformArithConstantOp(cst, valueAttrMap[cst]);
        } else if (auto loop = dyn_cast<scf::ForOp>(op)) {
          transformScfForOp(loop);
        } else if (auto store = dyn_cast<tt::StoreOp>(op)) {
          transformStoreOp(store);
        } else if (numResults != 0) {
          assert(numResults == 1);
          transformGenericOp(op, valueAttrMap);
        }
        return WalkResult::advance();
      });
    }

    /// adding module attributes
    mod->setAttr(tt::AttrNumWarpsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, numWarps.getValue())));
    mod->setAttr(tt::AttrNumThreadsPerWarp,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
    mod->setAttr(tt::AttrNumCTAsName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 1)));
    mod->setAttr(tt::AttrComputeCapabilityName,
                 IntegerAttr::get(i32Ty, llvm::APInt(32, 90)));
  }

  void transformGenericOp(Operation *op, DenseMap<Value, Attribute> &map) {
    Dialect *arithDialect = op->getContext()->getLoadedDialect("arith");
    Dialect *mathDialect = op->getContext()->getLoadedDialect("math");
    auto result = op->getResults()[0];
    // if already got
    if (map.count(result) != 0) {
      auto newType = addAttrToType(result.getType(), map[result]);
      result.setType(newType);
    }
    // get the attr by propagating
    else if (op->getDialect() == arithDialect ||
             op->getDialect() == mathDialect) {
      Attribute attr;
      for (auto operand : op->getOperands()) {
        if (auto type = dyn_cast<RankedTensorType>(operand.getType()))
          if (type.getEncoding())
            attr = type.getEncoding();
      }
      auto newType = addAttrToType(result.getType(), attr);
      result.setType(newType);
    }
  }

  // assume the below code sequence for now
  // %ptr = tt.make_tensor_ptr
  // tt.store %ptr, %value
  void transformStoreOp(tt::StoreOp op) {
    auto attr = cast<RankedTensorType>(op.getValue().getType()).getEncoding();
    auto makePtrOp = cast<tt::MakeTensorPtrOp>(op.getPtr().getDefiningOp());
    auto result = makePtrOp.getResult();
    auto newType = addAttrToType(result.getType(), attr);
    result.setType(newType.cast<tt::PointerType>());
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
    value = value.reshape(newType.cast<ShapedType>());
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

    // assume llvm/spirv dot size is 8x16
    //((n % 16) == 0 && (m % 8) == 0)
    auto maxWarpsX = n / 16;
    auto maxWarpsY = m / 8;
    auto root = std::sqrt(numWarps);
    auto numWarpsX = 1;
    if (n / 16 <= root)
      numWarpsX = n / 16;
    else if (n / 32 <= root)
      numWarpsX = n / 32;
    else if (n / 64 <= root)
      numWarpsX = n / 64;
    else
      numWarpsX = n / 128;
    warpsPerCTA[1] = numWarpsX;
    warpsPerCTA[0] = numWarps / warpsPerCTA[1];
    sizePerWarp[1] = n / warpsPerCTA[1];
    sizePerWarp[0] = m / warpsPerCTA[0];
    return {sizePerWarp, warpsPerCTA};
  }

  Workload matchLoopWorkload(scf::ForOp loop, LoopDotInfo loopDotInfo) {
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
      if (!info0.chainOpsA.empty() && // Q is loop invariant
          info0.chainOpsA[0].getDefiningOp()->isBeforeInBlock(loop) &&
          info0.advanceB && info1.advanceB) {
        SmallVector<OpFoldResult> rawOffsetsK = info0.advanceB.getOffsets();
        SmallVector<OpFoldResult> rawOffsetsV = info1.advanceB.getOffsets();
        auto offsetsK = *getConstantIntValues(rawOffsetsK);
        auto offsetsV = *getConstantIntValues(rawOffsetsV);
        if (offsetsK.size() == 2 && offsetsV.size() == 2 && offsetsK[0] == 0 &&
            offsetsV[1] == 0 && offsetsK[1] == offsetsV[0])
          return Workload::Attention;
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
    Dialect *arithDialect = val.getContext()->getLoadedDialect("arith");
    Dialect *mathDialect = val.getContext()->getLoadedDialect("math");
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
    Dialect *arithDialect = dot.getContext()->getLoadedDialect("arith");
    Dialect *mathDialect = dot.getContext()->getLoadedDialect("math");
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
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUWarpPass(int numWarps) {
  return std::make_unique<::ConvertTritonToTritonGPUWarp>(numWarps);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUWarpPass() {
  return std::make_unique<::ConvertTritonToTritonGPUWarp>();
}
