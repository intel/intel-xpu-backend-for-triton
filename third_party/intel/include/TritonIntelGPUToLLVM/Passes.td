#ifndef TRITONINTELGPU_CONVERSION_PASSES
#define TRITONINTELGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def IntelAllocateSharedMemory
    : Pass<"intel-allocate-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def ConvertTritonIntelGPUToLLVM
    : Pass<"convert-triton-intel-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert TritonIntelGPU to LLVM";

  let description = [{
    Lowers TritonIntelGPU to LLVM. Shared (work-group) memory is allocated
    statically in the module: `global_smem` is an internal array sized after the
    `ttg.shared` module attribute, which the Level Zero runtime allocates
    directly from the SPIR-V module, so no kernel argument is needed to pass the
    shared memory base.

    `dynamic-shared-memory` requests the legacy behavior instead: a zero sized
    array with external linkage, allocated by the runtime and passed in as an
    extra kernel argument (see `tritonintelgpu-rewrite-stack-ptr`).
  }];

  let options = [
    Option<"dynamicSharedMemory", "dynamic-shared-memory", "bool",
           /*default*/"false",
           "Request shared memory dynamically (as a kernel argument bound by "
           "the runtime) instead of allocating it statically in the module">
  ];

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
