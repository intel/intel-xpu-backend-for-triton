set(LLVM_TARGET_DEFINITIONS GPUToTritonGEN.td)
mlir_tablegen(GPUToTritonGEN.cpp.inc -gen-rewriters)
add_public_tablegen_target(GPUToTritonGENIncGen)

add_triton_library(GPUToTritonGEN
  GPUToTritonGENPass.cpp

  DEPENDS
  GPUToTritonGENConversionPassIncGen
  GPUToTritonGENIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRPass
)
