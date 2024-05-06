#ifndef TRITONINTELGPU_CONVERSION_PASSES
#define TRITONINTELGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def IntelDecomposeUnsupportedConversions
    : Pass<"intel-decompose-unsupported-conversions", "mlir::ModuleOp"> {
  let summary = "Decompose conversions that are not supported by TritonGPU -> LLVM";
}

def IntelAllocateSharedMemory
    : Pass<"intel-allocate-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";
}

def ConvertTritonIntelGPUToLLVM
    : Pass<"convert-triton-intel-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonIntelGPU to LLVM";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonGEN::TritonGENDialect"];
}

#endif // TRITONINTELGPU_CONVERSION_PASSES
