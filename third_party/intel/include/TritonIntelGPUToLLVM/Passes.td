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
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::math::MathDialect",
                           "mlir::gpu::GPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::tensor::TensorDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonGEN::TritonGENDialect"];
  let options = [
    Option<"advancedPath", "advanced_path",
           "bool", /*default*/"false",
           "enable advanced path">,
    Option<"oneMatrixPerLoadForBT", "one_matrix_per_load_for_bt",
           "bool", /*default*/"true",
           "Only load one DPAS operands per load for transposed B matrix">,
    Option<"useTileLoadLinearLayout", "use_tile_load_linear_layout",
           "bool", /*default*/"true",
           "Use linear layouts to generate the tile load sizes and offsets">
  ];
}

#endif // TRITONINTELGPU_CONVERSION_PASSES
