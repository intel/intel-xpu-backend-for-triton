add_mlir_translation_library(TritonGENToLLVMIRTranslation
  TritonGENToLLVMIRTranslation.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  TritonGENIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
