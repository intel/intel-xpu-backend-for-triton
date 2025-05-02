#ifndef TRITON_TRANSFORMS_UTILITY_H
#define TRITON_TRANSFORMS_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

Value getPredMask(RewriterBase &rewriter, Type typeLike, Value currentMask,
                  Value pred);

#endif // TRITON_TRANSFORMS_UTILITY_H
