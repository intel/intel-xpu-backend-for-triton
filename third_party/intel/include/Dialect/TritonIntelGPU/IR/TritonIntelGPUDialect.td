#ifndef TRITON_INTEL_GPU_DIALECT
#define TRITON_INTEL_GPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonIntelGPU_Dialect : Dialect {
  let name = "triton_intel_gpu";

  let cppNamespace = "::mlir::triton::gpu::intel";

  let description = [{
    Triton Intel GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
  ];

  let extraClassDeclaration = [{
    /// Get the name of the attribute used to indicate the target architecture.
    static constexpr ::llvm::StringLiteral getTargetAttrName() {
      return ::llvm::StringLiteral("triton_gpu.target");
    }

    /// Get the name of the attribute used to indicate the LTS driver.
    static ::llvm::StringLiteral getLTSAttrName() {
      return ::llvm::StringLiteral("triton_gpu.is_lts");
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
