#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonGPUToSPIRV : Pass<"convert-triton-gpu-to-spriv", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to SPIRV";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToSPIRVPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                            "mlir::math::MathDialect",
                            "mlir::gpu::GPUDialect",
                            "mlir::scf::SCFDialect",
                            "mlir::tensor::TensorDialect",
                            "mlir::triton::TritonDialect",
                            "mlir::triton::gpu::TritonGPUDialect",
                            "mlir::spirv::SPIRVDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
                "std::map<std::string, int>", /*default*/"std::map<std::string, int>{}",
                "device compute capability">
    ];
}

#endif
