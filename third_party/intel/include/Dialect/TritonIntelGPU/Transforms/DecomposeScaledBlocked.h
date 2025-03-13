#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::gpu::intel {

void populateDecomposeScaledBlockedPatterns(mlir::RewritePatternSet &patterns,
                                            int benefit);

} // namespace mlir::triton::gpu::intel
