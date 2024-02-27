#ifndef NVGPUINTEL_CONVERSION_PASSES
#define NVGPUINTEL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertNVGPUIntelToLLVM : Pass<"convert-nv-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert NVGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertNVGPUIntelToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::nvgpu::NVGPUDialect"];
}

#endif
