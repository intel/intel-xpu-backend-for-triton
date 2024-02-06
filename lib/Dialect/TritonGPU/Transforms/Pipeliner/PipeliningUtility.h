#ifndef TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_

#include "mlir/Dialect/SCF/IR/SCF.h"
#include <vector>

namespace mlir {
namespace triton {

/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

/// Collect ssa dependencies of `op` in `deps`. if `includeArg` is true,
/// continue looking through loop block arguments.
void addDep(Operation *op, DenseSet<Operation *> &deps, bool includeArg = true,
            DenseSet<Operation *> *filter = nullptr);

/// Add operations from `forOp` into a pipeline schedule with the the given
/// `stage` when filter is true. This will add operation in the original loop
/// order.
void addOps(scf::ForOp forOp, int stage,
            std::vector<std::pair<Operation *, unsigned>> &schedule,
            std::function<bool(Operation *)> filter);
} // namespace triton
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_PIPELINER_PIPELINING_UTILITY_H_
