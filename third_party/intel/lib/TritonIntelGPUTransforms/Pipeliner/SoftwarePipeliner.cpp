#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Pipeliner/Schedule.h"

#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"

using namespace mlir;
namespace ttgi = mlir::triton::gpu::intel;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUPIPELINE
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

// Return true if the preconditions for pipelining the loop are met.
static bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (isa<LoopLikeOpInterface>(op) && forOp.getOperation() != op)
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

// Prove scalar control flow, including read-only metadata lookup loops.
class WorkgroupUniformity {
public:
  explicit WorkgroupUniformity(triton::FuncOp kernel) : kernel(kernel) {}

  bool isUniform(Value value) {
    if (!value.getType().isIntOrIndex() &&
        !isa<triton::PointerType>(value.getType()))
      return false;
    auto found = known.find(value);
    if (found != known.end())
      return found->second;
    known[value] = false;
    bool uniform = false;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      uniform = arg.getOwner() == &kernel.getBody().front();
    } else if (Operation *def = value.getDefiningOp()) {
      if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def)) {
        uniform = true;
      } else if (isa<arith::ArithDialect>(def->getDialect()) ||
                 isa<triton::AddPtrOp>(def)) {
        uniform = allUniform(def->getOperands());
      } else if (auto load = dyn_cast<triton::LoadOp>(def)) {
        uniform = !load.getIsVolatile() &&
                  (!load.getMask() || load.getOther()) &&
                  isReadOnlyEntryLoad(load) && allUniform(load->getOperands());
      } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
        unsigned index = cast<OpResult>(value).getResultNumber();
        uniform = isUniform(ifOp.getCondition()) &&
                  isUniform(ifOp.thenYield().getOperand(index)) &&
                  isUniform(ifOp.elseYield().getOperand(index));
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
        uniform = isUniformWhile(whileOp);
      }
    }
    known[value] = uniform;
    return uniform;
  }

  bool hasUniformBranches() {
    for (Block &block : kernel.getBody()) {
      Operation *term = block.getTerminator();
      if (auto branch = dyn_cast<cf::CondBranchOp>(term)) {
        if (!isUniform(branch.getCondition()))
          return false;
      } else if (!isa<cf::BranchOp, triton::ReturnOp>(term)) {
        return false;
      }
    }
    return true;
  }

private:
  bool allUniform(ValueRange values) {
    return llvm::all_of(values, [&](Value v) { return isUniform(v); });
  }

  static bool isReadOnly(Operation *op) {
    auto effects = getEffectsRecursively(op);
    return effects && llvm::all_of(*effects, [](const auto &effect) {
             return isa<MemoryEffects::Read>(effect.getEffect());
           });
  }

  bool isReadOnlyEntryLoad(triton::LoadOp load) {
    Operation *top = load;
    while (top->getParentOp() != kernel) {
      top = top->getParentOp();
      if (!isa<scf::IfOp, scf::WhileOp>(top) || !isReadOnly(top))
        return false;
    }
    Block &entry = kernel.getBody().front();
    if (top->getBlock() != &entry)
      return false;
    // Do not infer uniform metadata from loads after writes or opaque effects.
    for (Operation &op : entry) {
      if (&op == top)
        return true;
      if (!isReadOnly(&op))
        return false;
    }
    return false;
  }

  bool isUniformWhile(scf::WhileOp loop) {
    if (!llvm::hasSingleElement(loop.getBefore()) ||
        !llvm::hasSingleElement(loop.getAfter()) ||
        !allUniform(loop.getInits()) || !isReadOnly(loop))
      return false;
    // Prove the induction in a private cache; failed hypotheses must not leak.
    WorkgroupUniformity body(*this);
    for (BlockArgument arg : loop.getBeforeArguments())
      body.known[arg] = true;
    auto condition = loop.getConditionOp();
    if (!body.isUniform(condition.getCondition()) ||
        !body.allUniform(condition.getArgs()))
      return false;
    for (BlockArgument arg : loop.getAfterArguments())
      body.known[arg] = true;
    return body.allUniform(loop.getYieldOp().getOperands());
  }

  triton::FuncOp kernel;
  DenseMap<Value, bool> known;
};

static bool isWorkgroupBarrierCandidate(scf::ForOp loop) {
  auto module = loop->getParentOfType<ModuleOp>();
  if (!module->hasAttr(ttgi::TritonIntelGPUDialect::
                           getSupportSplitWorkGroupBarrierAttrName()))
    return false;
  auto kernel = dyn_cast<triton::FuncOp>(loop->getParentOp());
  if (!kernel || !kernel.isPublic())
    return false;
  // Opaque code could leave an unnamed split barrier outstanding across the
  // loop.
  if (kernel
          .walk([](Operation *op) {
            return isa<triton::CallOp, triton::ElementwiseInlineAsmOp>(op)
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  WorkgroupUniformity uniformity(kernel);
  return uniformity.hasUniformBranches() &&
         uniformity.isUniform(loop.getLowerBound()) &&
         uniformity.isUniform(loop.getUpperBound()) &&
         uniformity.isUniform(loop.getStep());
}

static void pipelineLoop(scf::ForOp forOp, int numStages, bool useBarrier) {
  mlir::scf::PipeliningOption options;
  if (!preCondition(forOp))
    return;

  bool foundSchedule =
      ttgi::preProcessLoopAndGetSchedule(forOp, numStages, options);
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::scf::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return;

  scf::ForOp loop = (*newForOp);
  if (useBarrier) {
    OpBuilder b(loop);
    Location loc = loop.getLoc();
    b.setInsertionPointToStart(loop.getBody());
    auto candidate =
        isWorkgroupBarrierCandidate(loop) ? b.getUnitAttr() : UnitAttr{};
    auto bData =
        triton::TritonGEN::SplitBarrierArriveOp::create(b, loc, candidate);
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    b.setInsertionPoint(yield);
    triton::TritonGEN::SplitBarrierWaitOp::create(b, loc, bData);
  }
}

namespace {
struct IntelGPUPipelinePass
    : public triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
          IntelGPUPipelinePass> {

  using triton::gpu::intel::impl::TritonIntelGPUPipelineBase<
      IntelGPUPipelinePass>::TritonIntelGPUPipelineBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    if (!m->hasAttr(ttgi::TritonIntelGPUDialect::getSupport2DBlockIOAttrName()))
      return;

    if (numStages <= 1)
      return;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });

    for (scf::ForOp forOp : loops)
      pipelineLoop(forOp, numStages, useBarrier);
  }
};
} // anonymous namespace
