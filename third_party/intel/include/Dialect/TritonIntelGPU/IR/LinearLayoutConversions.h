// Conversions from TritonIntelGPU DpasEncodingAttr to LinearLayout.

#ifndef TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::gpu {

// DPAS operand A: opIdx=0
// DPAS operand B: opIdx=1
// DPAS operand C (default): opIdx=2
// Operand A and B conversion are not used yet
std::optional<LinearLayout> DPAStoLinearLayout(ArrayRef<int64_t> shape,
                                               Attribute layout,
                                               unsigned opIdx = 2);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H
