#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"
#include "triton/Tools/Sys/GetEnv.hpp"

using namespace mlir;
using namespace mlir::triton;

TritonIntelGPUToLLVMTypeConverter::TritonIntelGPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : TritonGPUToLLVMTypeConverter(ctx, option, analysis) {
  // The following type conversions have been registered by the base class (in
  // the constructor)
  if (mlir::triton::tools::getBoolEnv("TRITON_INTEL_ENABLE_BLOCK_PTR")) {
    // tt::pointer to v2i32
    addConversion([&](PointerType type) -> std::optional<Type> {
      if (isa<RankedTensorType>(type.getPointeeType())) {
        auto i32Type = mlir::IntegerType::get(type.getContext(), 32);
        return mlir::VectorType::get(2, i32Type);
      }
      return LLVM::LLVMPointerType::get(type.getContext(),
                                        type.getAddressSpace());
    });

    // tensor type is flattened and divided by 16(subgroupSize)
    addConversion([&](mlir::RankedTensorType type) -> mlir::Type {
      return mlir::VectorType::get(type.getNumElements() / 16,
                                   type.getElementType());
    });
  }
}
