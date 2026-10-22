#include "Conversion/ProtonGPUToLLVM/ProtonIntelGPUToLLVM/TargetInfo.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENMemorySpace.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/lib/Utils/LLVMIntr.h"
#include "intel/lib/Utils/Mangling.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton::proton::gpu::Intel {

Value TargetInfo::clock(ConversionPatternRewriter &rewriter, Location loc,
                        bool isClock64) const {
  auto mod = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  bool isLTS = mod->hasAttr(
      triton::gpu::intel::TritonIntelGPUDialect::getIsLTSAttrName());
  std::string funcName = isLTS ? "__builtin_IB_read_cycle_counter"
                               : mlir::triton::gpu::intel::mangle(
                                     "__spirv_ReadClockKHR_Rulong", {i32_ty});

  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Type> argTypes;
  SmallVector<Value> args;
  if (!isLTS) {
    argTypes.push_back(i32_ty);
    args.push_back(b.i32_val(/*Device=*/1));
  }
  auto callOp = mlir::triton::gpu::intel::createDeviceFunctionCall(
      rewriter, funcName, /*retType=*/i64_ty, argTypes, args,
      /*paramAttrs=*/{}, mlir::triton::gpu::intel::noUnwindWillReturnAttrs);
  // `createDeviceFunctionCall` always stamps `UnknownLoc`; reattach the real
  // location so e.g. line-info-sensitive tooling can still find this call.
  callOp->setLoc(loc);
  Value ticks = callOp.getResult();
  if (!isClock64)
    ticks = LLVM::TruncOp::create(rewriter, loc, i32_ty, ticks);
  return ticks;
}

Value TargetInfo::globalTime(ConversionPatternRewriter &rewriter,
                             Location loc) const {
  return getTritonTargetInfo().getGlobalTimer(rewriter, loc);
}

Value TargetInfo::processorId(ConversionPatternRewriter &rewriter,
                              Location loc) const {
  // `mangle()` doesn't special-case an empty parameter list: Itanium mangling
  // still requires a trailing `v` ("void") for a zero-argument function.
  std::string funcName =
      mlir::triton::gpu::intel::mangle("intel_get_hw_thread_id", {}) + "v";
  auto callOp = mlir::triton::gpu::intel::createDeviceFunctionCall(
      rewriter, funcName, /*retType=*/i32_ty, /*argTypes=*/{}, /*args=*/{},
      /*paramAttrs=*/{}, mlir::triton::gpu::intel::noUnwindWillReturnAttrs);
  callOp->setLoc(loc);
  return callOp.getResult();
}

int TargetInfo::getAddressSpace(Attribute addressSpace) const {
  int spaceId = 0;
  if (mlir::isa<triton::gpu::SharedMemorySpaceAttr>(addressSpace)) {
    spaceId = TritonGEN::TritonGENMemorySpace::kWorkgroup;
  } else if (mlir::isa<proton::gpu::GlobalMemorySpaceAttr>(addressSpace)) {
    spaceId = TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  } else {
    llvm::report_fatal_error("Only support SharedMemorySpace, "
                             "and GlobalMemorySpace for now");
  }
  return spaceId;
}

unsigned TargetInfo::getPtrAddressSpace(triton::PtrAddrSpace space) const {
  switch (space) {
  case triton::PtrAddrSpace::Global:
    return TritonGEN::TritonGENMemorySpace::kCrossWorkgroup;
  case triton::PtrAddrSpace::Constant:
    return TritonGEN::TritonGENMemorySpace::kUniformConstant;
  case triton::PtrAddrSpace::Descriptor:
    return TritonGEN::TritonGENMemorySpace::kGeneric;
  }
  llvm_unreachable("unknown PtrAddrSpace");
}

int TargetInfo::getIndexPtrAddrSpace() const {
  // The internal per-thread buffer index is private to each work-item.
  return TritonGEN::TritonGENMemorySpace::kFunction;
}

} // namespace mlir::triton::proton::gpu::Intel
