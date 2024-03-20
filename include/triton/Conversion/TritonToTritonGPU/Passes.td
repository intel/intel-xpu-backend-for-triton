#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPU: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,

       Option<"threadsPerWarp", "threads-per-warp",
              "int32_t", /*default*/"32",
              "number of threads per warp">,
        Option<"numCTAs", "num-ctas",
              "int32_t", /*default*/"1",
              "number of ctas in a cga">,
        Option<"computeCapability", "compute-capability",
              "int32_t", /*default*/"80",
              "compute capability">
   ];
}

def ConvertTritonToTritonGPUWarp: Pass<"convert-triton-to-tritongpu-warp", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU with warp distribute annotation";
    let description = [{
        This pass first analyze the kernel's workload pattern (elementwise/reduction/gemm/attention),
        and then figure out the best layout for key/anchor operation (dot in gemm case).
        Afterwards, we can get all other operation’s layout through def/use chain.
        Finally, each tensor operation is annotated with layout attribute describing what each warp should do.
    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUWarpPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "unsigned", /*default*/"4",
              "number of warps">,
   ];
}


#endif
