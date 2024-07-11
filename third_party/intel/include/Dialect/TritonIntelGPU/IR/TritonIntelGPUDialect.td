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

    /// Get the name of the attribute used to indicate whether subgroup 2D block
    /// operations (e.g., 2D block read/write) are available.
    static llvm::StringRef getSupportSG2DBlockAttrName() {
      return "triton_intel_gpu.support_sg_2d_block";
    }

    /// Get the name of the attribute used to indicate whether the DPAS
    /// instruction is available.
    static llvm::StringRef getSupportDPASAttrName() {
      return "triton_intel_gpu.support_dpas";
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
