#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_BASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/TypeConverter.h"
#include "intel/lib/TritonIntelGPUToLLVM/Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::delinearize;
using ::mlir::LLVM::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;

typedef DenseMap<Operation *, triton::MakeTensorPtrOp> TensorPtrMapT;

class ConvertTritonGPUOpToLLVMPatternBase {
public:
  explicit ConvertTritonGPUOpToLLVMPatternBase(
      const LLVMTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  const LLVMTypeConverter *getTypeConverter() const { return converter; }

protected:
  const LLVMTypeConverter *converter;
};

template <typename SourceOp>
class ConvertTritonGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToLLVMPattern(
      const LLVMTypeConverter &typeConverter, PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

protected:
  TritonIntelGPUToLLVMTypeConverter *getTypeConverter() const {
    const LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (TritonIntelGPUToLLVMTypeConverter *)ret;
  }
};

#endif // TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_BASE_H
