#pragma once
#include "intel/include/Dialect/Triton/Transforms/Passes.h"
#include "intel/include/Dialect/TritonGEN/IR/TritonGENDialect.h"
#include "intel/include/Dialect/TritonIntelGPU/IR/Dialect.h"
#include "intel/include/Dialect/TritonIntelGPU/Transforms/Passes.h"
#include "intel/include/GPUToTritonGEN/Passes.h"
#include "intel/include/TritonAnnotateModule/Passes.h"
#include "intel/include/TritonGENToLLVM/Passes.h"
#include "intel/include/TritonIntelGPUToLLVM/Passes.h"
#include "intel/include/TritonRaiseBlockPointer/Passes.h"
#include "intel/include/TritonToTritonGPUWarp/Passes.h"

#include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "amd/include/TritonAMDGPUTransforms/Passes.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// Below headers will allow registration to ROCm passes
#include "TritonAMDGPUToLLVM/Passes.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "nvidia/include/NVGPUToLLVM/Passes.h"
#include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"

namespace mlir {
namespace test {
namespace intel {
void registerTestAxisInfoPass();
}

void registerTestAliasPass();
void registerTestAlignmentPass();
void registerTestAllocationPass();
void registerTestLivenessPass();
void registerTestMembarPass();
void registerTestTritonAMDGPURangeAnalysis();
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::registerTritonNvidiaGPUPasses();
  mlir::test::intel::registerTestAxisInfoPass();
  mlir::test::registerTestAliasPass();
  mlir::test::registerTestAlignmentPass();
  mlir::test::registerTestAllocationPass();
  mlir::test::registerTestLivenessPass();
  mlir::test::registerTestMembarPass();
  mlir::test::registerTestTritonAMDGPURangeAnalysis();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::intel::registerConvertTritonToTritonGPUWarpPass();
  mlir::triton::intel::registerTritonIntelTensorDescToBlockPointer();
  mlir::triton::intel::registerTritonIntelRemoveMasks();
  mlir::triton::intel::registerTritonRaiseBlockPointer();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerConvertWarpSpecializeToLLVM();
  mlir::triton::registerConvertTritonGPUToLLVMPass();
  mlir::triton::registerConvertNVGPUToLLVMPass();
  mlir::registerLLVMDIScope();
  mlir::triton::gpu::intel::registerTritonAnnotateModulePass();
  mlir::triton::gpu::intel::registerTritonIntelGPUPasses();
  mlir::triton::gpu::intel::registerTritonIntelGPUToLLVMPasses();
  mlir::triton::registerConvertGPUToTritonGEN();
  mlir::triton::registerConvertTritonGENToLLVM();
  mlir::triton::registerTritonGENToLLVMPasses();

  // TritonAMDGPUToLLVM passes
  mlir::triton::registerConvertTritonAMDGPUToLLVM();
  mlir::triton::registerConvertBuiltinFuncToLLVM();
  mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  mlir::registerTritonAMDGPUAccelerateMatmul();
  mlir::registerTritonAMDGPUOptimizeEpilogue();
  mlir::registerTritonAMDGPUHoistLayoutConversions();
  mlir::registerTritonAMDGPUReorderInstructions();
  mlir::registerTritonAMDGPUBlockPingpong();
  mlir::registerTritonAMDGPUStreamPipeline();
  mlir::registerTritonAMDGPUCanonicalizePointers();
  mlir::registerTritonAMDGPUConvertToBufferOps();
  mlir::registerTritonAMDGPUInThreadTranspose();
  mlir::triton::registerTritonAMDGPUInsertInstructionSchedHints();
  mlir::triton::registerTritonAMDGPULowerInstructionSchedHints();

  registry
      .insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
              mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,
              mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
              mlir::NVVM::NVVMDialect, mlir::triton::nvgpu::NVGPUDialect,
              mlir::triton::amdgpu::TritonAMDGPUDialect,
              mlir::triton::proton::ProtonDialect, mlir::ROCDL::ROCDLDialect,
              mlir::triton::gpu::intel::TritonIntelGPUDialect,
              mlir::triton::TritonGEN::TritonGENDialect>();
}
