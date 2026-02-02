add_mlir_translation_library(TritonGENToLLVMIRTranslation
  TritonGENToLLVMIRTranslation.cpp

  DEPENDS
  TritonGENTableGen
  TritonGENAttrDefsIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  TritonGENIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
)
