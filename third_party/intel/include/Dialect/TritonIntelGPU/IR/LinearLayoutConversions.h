// Conversions from TritonIntelGPU DpasEncodingAttr to LinearLayout.

#ifndef TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "intel/include/Dialect/TritonIntelGPU/IR/Attributes.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::gpu {

std::optional<LinearLayout> DPAStoLinearLayout(ArrayRef<int64_t> shape,
                                               Attribute layout);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONINTELGPU_IR_LINEARLAYOUTCONVERSIONS_H
