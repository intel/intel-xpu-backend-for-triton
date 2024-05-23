#ifndef TRITONGPUWARP_CONVERSION_PASSES
#define TRITONGPUWARP_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPUWarp: Pass<"convert-triton-to-tritongpu-warp", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU with warp distribute annotation";
    let description = [{
        This pass first analyze the kernel's workload pattern (elementwise/reduction/gemm/attention),
        and then figure out the best layout for key/anchor operation (dot in gemm case).
        Afterwards, we can get all other operation’s layout through def/use chain.
        Finally, each tensor operation is annotated with layout attribute describing what each warp should do.
    }];

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
