#pragma once
#include "triton/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#ifdef USE_ROCM
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"
#endif
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

// FIXME
#include "intel/include/NVGPUIntelToLLVM/Passes.h"
#include "intel/include/TritonGENToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {
namespace test {
void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::registerTritonGPUPasses();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::registerTritonIntelGPUPasses();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestMembarPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerConvertTritonToTritonGPUWarpPass();
  mlir::triton::registerDecomposeUnsupportedConversionsPass();
  mlir::triton::registerAllocateSharedMemoryPass();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::registerLLVMDIScope();
  // Intel passes
  mlir::triton::registerIntelDecomposeUnsupportedConversionsPass();
  mlir::triton::registerIntelAllocateSharedMemoryPass();
  mlir::triton::registerConvertTritonIntelGPUToLLVMPass();
  mlir::triton::registerConvertTritonGENToLLVM();
  mlir::triton::registerConvertNVGPUIntelToLLVMPass();
  mlir::triton::registerTritonGENToLLVMPasses();

#ifdef USE_ROCM
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerDecomposeUnsupportedAMDConversionsPass();

  // TODO: Uncomment when fixed undefined symbols and
  // remove section below
  // List of undefined symbols:
  // createTritonAMDGPUCoalesce is not defined
  // createTritonAMDGPUOptimizeDotOperands is not defined
  // createTritonAMDGPUPipeline is not defined
  // createTritonAMDGPUPrefetch is not defined

  // mlir::registerTritonAMDGPUPasses();

  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUDecomposeConversions();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPURemoveLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUStreamPipeline();

#endif // USE_ROCM

  // TODO: register Triton & TritonGPU passes
  registry.insert<
      mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
      mlir::triton::gpu::intel::TritonIntelGPUDialect,
      mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
      mlir::triton::gpu::TritonGPUDialect,
      mlir::triton::TritonGEN::TritonGENDialect, mlir::math::MathDialect,
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, mlir::gpu::GPUDialect,
      mlir::LLVM::LLVMDialect, mlir::NVVM::NVVMDialect,
      mlir::triton::nvgpu::NVGPUDialect, mlir::ROCDL::ROCDLDialect>();
}
