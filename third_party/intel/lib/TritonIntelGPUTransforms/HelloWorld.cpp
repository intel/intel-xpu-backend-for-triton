#include "Dialect/TritonIntelGPU/IR/Attributes.h"
#include "Dialect/TritonIntelGPU/Transforms/Utility.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"


using namespace mlir;

namespace mlir::triton::gpu::intel {
#define GEN_PASS_DEF_TRITONINTELGPUHELLOWORLD
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu::intel


class TritonIntelGPUHelloWorldPass
    : public triton::gpu::intel::impl::TritonIntelGPUHelloWorldBase<
          TritonIntelGPUHelloWorldPass> {
public:
  using triton::gpu::intel::impl::TritonIntelGPUHelloWorldBase<
      TritonIntelGPUHelloWorldPass>::TritonIntelGPUHelloWorldBase;

  void runOnOperation() override {
    llvm::outs() << "Hello from XPU TTGIR pass!\n";
  }
};

