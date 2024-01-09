#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TYPECONVERTER_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_TYPECONVERTER_H

#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;

class TritonGPUToSPIRVTypeConverter : public SPIRVTypeConverter {
public:
  using TypeConverter::convertType;

  TritonGPUToSPIRVTypeConverter(spirv::TargetEnvAttr &targetAttr,
                                SPIRVConversionOptions &option);

  Type getElementTypeForStruct(RankedTensorType type);
  Type convertTritonPointerType(triton::PointerType type);

  Value packLLElements(Location loc, ValueRange resultVals,
                       ConversionPatternRewriter &rewriter, Type type);

  SmallVector<Value> unpackLLElements(Location loc, Value spirvStruct,
                                      ConversionPatternRewriter &rewriter,
                                      Type type);

  Type convertTritonTensorType(RankedTensorType type);
};

#endif
