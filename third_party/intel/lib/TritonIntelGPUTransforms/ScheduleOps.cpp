#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUSCHEDULEOPS
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel

namespace ttgi = mlir::triton::gpu::intel;
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

class ScheduleOpsPass
    : public intel::impl::TritonIntelGPUScheduleOpsBase<ScheduleOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp mod = getOperation();

    mod.walk([&](scf::ForOp forOp) {
      llvm::SmallVector<triton::LoadOp> loadOps;
      for (Operation &opInFor : forOp) {
        if (LoadOp loadOp = dyn_cast<triton::LoadOp>(opInFor)) {
          if (loadOp->hasOneUse()) {
            auto users = loadOp->getUsers();
            DotOp dotOp = dyn_cast<triton::DotOp>(*(users.begin()));
            if (dotOp)
              loadOps.push_back(loadOp);
          }
        }
      }

      for (LoadOp &loadOp : loadOps) {
        auto users = loadOp->getUsers();
        loadOp->moveBefore(*users.begin());
      }
    });
  }
};

} // namespace
