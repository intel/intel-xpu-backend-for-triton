#ifndef PROTONGPU_TO_LLVM_INTEL_PATTERN_PROTONGPU_OP_TO_LLVM_H
#define PROTONGPU_TO_LLVM_INTEL_PATTERN_PROTONGPU_OP_TO_LLVM_H

#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::proton::gpu::Intel {
void populateProtonGPUOpIntelPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      const TargetInfo &targetInfo,
                                      PatternBenefit benefit);
} // namespace mlir::triton::proton::gpu::Intel

#endif // PROTONGPU_TO_LLVM_INTEL_PATTERN_PROTONGPU_OP_TO_LLVM_H
