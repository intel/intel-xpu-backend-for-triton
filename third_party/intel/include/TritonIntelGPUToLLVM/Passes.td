#ifndef TRITONINTELGPU_CONVERSION_PASSES
#define TRITONINTELGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def IntelDecomposeUnsupportedConversions : Pass<"intel-decompose-unsupported-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose conversions that are not supported by TritonGPU -> LLVM";
    let constructor = "mlir::triton::gpu::createIntelDecomposeUnsupportedConversionsPass()";
}

def IntelAllocateSharedMemory : Pass<"intel-allocate-shared-memory", "mlir::ModuleOp"> {
    let summary = "Add metadata for shared memory allocation";
    let constructor = "mlir::triton::gpu::createIntelAllocateSharedMemoryPass()";
}

def ConvertTritonIntelGPUToLLVM : Pass<"convert-triton-intel-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonIntelGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonIntelGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::triton::TritonGEN::TritonGENDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

#endif
