#ifndef PROTONGPU_TO_LLVM_TARGETINFO_INTEL_H
#define PROTONGPU_TO_LLVM_TARGETINFO_INTEL_H

#include "Conversion/ProtonGPUToLLVM/TargetInfoBase.h"
#include "intel/include/TritonIntelGPUToLLVM/TargetInfo.h"

namespace mlir::triton::proton::gpu::Intel {
class TargetInfo : public mlir::triton::proton::gpu::TargetInfoBase {
public:
  explicit TargetInfo(const mlir::triton::intel::TargetInfo &helper)
      : mlir::triton::proton::gpu::TargetInfoBase(helper) {}

  const mlir::triton::intel::TargetInfo &getTritonTargetInfo() const override {
    return static_cast<const mlir::triton::intel::TargetInfo &>(helper);
  }

  Value clock(ConversionPatternRewriter &rewriter, Location loc,
              bool isClock64) const override;

  Value globalTime(ConversionPatternRewriter &rewriter,
                   Location loc) const override;

  Value processorId(ConversionPatternRewriter &rewriter,
                    Location loc) const override;

  int getAddressSpace(Attribute addressSpace) const override;

  unsigned getPtrAddressSpace(triton::PtrAddrSpace space) const override;

  int getIndexPtrAddrSpace() const override;

  ~TargetInfo() = default;
};
} // namespace mlir::triton::proton::gpu::Intel

#endif // PROTONGPU_TO_LLVM_TARGETINFO_INTEL_H
