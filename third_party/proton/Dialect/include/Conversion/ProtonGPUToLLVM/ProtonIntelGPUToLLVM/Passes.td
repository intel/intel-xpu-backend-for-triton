#ifndef PROTONINTELGPU_TO_LLVM_PASSES
#define PROTONINTELGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonIntelGPUToLLVM : Pass<"convert-proton-intel-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using Intel-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonIntelGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];
}

#endif // PROTONINTELGPU_TO_LLVM_PASSES
