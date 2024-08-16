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
    /// Get the name of the attribute used to indicate whether the BF16 conversion instruction is available.
    static constexpr llvm::StringRef getSupportBF16ConversionAttrName() {
      return "triton_intel_gpu.support_bf16_conversion";
    }
  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
