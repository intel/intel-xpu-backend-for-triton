#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonIntelGPU/IR/Ops.cpp.inc"

namespace mlir {
namespace triton {
namespace gpu {
namespace intel {

void PrefetchCacheOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // The write effect to protect optimization to move the ops away.
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

} // namespace intel
} // namespace gpu
} // namespace triton
} // namespace mlir
