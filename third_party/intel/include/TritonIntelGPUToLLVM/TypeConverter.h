#ifndef TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H

#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

class TritonIntelGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  using TypeConverter::convertType;

  TritonIntelGPUToLLVMTypeConverter(
      MLIRContext *ctx, LowerToLLVMOptions &option,
      const DataLayoutAnalysis *analysis = nullptr);
};

#endif // TRITON_CONVERSION_TRITONINTELGPUTOLLVM_TYPECONVERTER_H
