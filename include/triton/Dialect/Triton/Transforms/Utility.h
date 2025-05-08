#ifndef TRITON_TRANSFORMS_UTILITY_H
#define TRITON_TRANSFORMS_UTILITY_H

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

namespace mlir::triton {

triton::MakeTensorPtrOp getMakeTensorPtrOp(Value v);

} // namespace mlir::triton

#endif // TRITON_TRANSFORMS_UTILITY_H
