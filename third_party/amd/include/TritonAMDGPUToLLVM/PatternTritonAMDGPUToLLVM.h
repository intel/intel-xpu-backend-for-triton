#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PATTERNTRITONAMDGPUTOLLVM_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PATTERNTRITONAMDGPUTOLLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::AMD {

void populateExtractSliceOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter, mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit);

}

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_PATTERNTRITONAMDGPUTOLLVM_H_
