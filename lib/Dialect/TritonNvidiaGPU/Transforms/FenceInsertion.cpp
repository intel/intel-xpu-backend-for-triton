#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered acorss genric and async proxy.
//
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

struct FenceInsertionPass
    : public TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  FenceInsertionPass() = default;
  FenceInsertionPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    if (::triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](Operation *op) {
      if (isa<tt::DotOp, ttng::DotAsyncOp>(op)) {
        OpBuilder builder(op);
        auto a = op->getOperand(0);
        auto b = op->getOperand(1);
        auto mmaEncoding = op->getResult(0)
                               .getType()
                               .cast<RankedTensorType>()
                               .getEncoding()
                               .dyn_cast<ttg::NvidiaMmaEncodingAttr>();
        auto isHopperEncoding = mmaEncoding && mmaEncoding.isHopper();
        if (isHopperEncoding &&
            (dependOnSharedEncOperand(a) || dependOnSharedEncOperand(b))) {
          builder.create<ttng::FenceAsyncSharedOp>(op->getLoc(),
                                                   false /*bCluster*/);
        }
      }
    });
  }

private:
  bool dependOnSharedEncOperand(Value operand) {
    static DenseSet<std::pair<Operation *, unsigned>> trace;
    auto op = operand.getDefiningOp();
    // avoid redundant insertion
    if (op && isa<tt::DotOp, ttng::DotAsyncOp>(op))
      return false;
    // reach convertlayout
    if (op && isa<ttg::ConvertLayoutOp>(op) && ttg::hasSharedEncoding(operand))
      return true;
    // root and not BlockArgument
    if (!op && !isa<BlockArgument>(operand))
      return false;
    // op and not BlockArgument
    if (op && !isa<BlockArgument>(operand)) {
      for (auto v : op->getOperands()) {
        if (dependOnSharedEncOperand(v))
          return true;
      }
    }
    // reach BlockArgument
    // TODO: support other scf ops, IfOp, WhileOp, etc.
    if (BlockArgument arg = dyn_cast<BlockArgument>(operand)) {
      unsigned argNum = arg.getArgNumber();
      Operation *argOwner = arg.getOwner()->getParentOp();
      // suport ForOp only
      if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
        // prologue
        auto iterOperands = forOp.getInitArgs();
        if (argNum == 0)
          return false;
        if (dependOnSharedEncOperand(iterOperands[argNum - 1]))
          return true;
        // yield
        auto yieldOp = forOp.getBody()->getTerminator();
        Value v = yieldOp->getOperand(argNum - 1);
        auto entry = std::make_pair<Operation *, unsigned>(std::move(yieldOp),
                                                           std::move(argNum));
        // avoid cyclic
        if (trace.contains(entry))
          return false;
        else
          trace.insert(entry);

        if (dependOnSharedEncOperand(v))
          return true;
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(argOwner)) {
        assert(false && "FenceInsertionPass does not supported WhileOp");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(argOwner)) {
        assert(false && "FenceInsertionPass does not supported IfOp");
      }
    }
    return false;
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUFenceInsertionPass(int computeCapability) {
  return std::make_unique<FenceInsertionPass>(computeCapability);
}
