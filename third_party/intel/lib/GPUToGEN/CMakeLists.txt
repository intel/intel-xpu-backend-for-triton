set(LLVM_TARGET_DEFINITIONS GPUToGEN.td)
mlir_tablegen(GPUToGEN.cpp.inc -gen-rewriters)
add_public_tablegen_target(GPUToGENIncGen)

add_triton_library(GPUToGEN
  GPUToGENPass.cpp

  DEPENDS
  GPUToGENConversionPassIncGen
  GPUToGENIncGen

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
