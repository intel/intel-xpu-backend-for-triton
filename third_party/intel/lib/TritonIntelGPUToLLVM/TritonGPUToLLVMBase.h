#ifndef TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_BASE_H
#define TRITON_CONVERSION_TRITONINTELGPU_TO_LLVM_BASE_H

// TODO: refactor so that it doesn't fail if Allocation.h
// is included after utility.h (due to conflict in `store` macro
// and <atomic>
#include "triton/Analysis/Allocation.h"

#include "TypeConverter.h"
//
#include "TritonIntelGPUToLLVM/Passes.h"
#include "Utility.h"
#include "mlir/IR/TypeUtilities.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <set>
#include <type_traits>

#define DEBUG_TYPE "ttgpu_to_llvm"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::utils::delinearize;
using ::mlir::LLVM::utils::SharedMemoryObject;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::CTALayoutAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SliceEncodingAttr;
using ::mlir::triton::gpu::intel::DpasEncodingAttr;
namespace ttng = ::mlir::triton::nvidia_gpu;

typedef DenseMap<Operation *, triton::MakeTensorPtrOp> TensorPtrMapT;

class ConvertTritonGPUOpToLLVMPatternBase {
public:
  explicit ConvertTritonGPUOpToLLVMPatternBase(
      TritonIntelGPUToLLVMTypeConverter &typeConverter)
      : converter(&typeConverter) {}

  TritonIntelGPUToLLVMTypeConverter *getTypeConverter() const {
    return converter;
  }

protected:
  TritonIntelGPUToLLVMTypeConverter *converter;
};

template <typename SourceOp>
class ConvertTritonGPUOpToLLVMPattern
    : public ConvertOpToLLVMPattern<SourceOp>,
      public ConvertTritonGPUOpToLLVMPatternBase {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ConvertTritonGPUOpToLLVMPattern(
      TritonIntelGPUToLLVMTypeConverter &typeConverter,
      PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        ConvertTritonGPUOpToLLVMPatternBase(typeConverter) {}

protected:
  TritonIntelGPUToLLVMTypeConverter *getTypeConverter() const {
    LLVMTypeConverter *ret =
        ((ConvertTritonGPUOpToLLVMPatternBase *)this)->getTypeConverter();
    return (TritonIntelGPUToLLVMTypeConverter *)ret;
  }
};

#endif
