add_subdirectory(Dialect)

add_mlir_translation_library(PostProcessLLVMIR
  LICM.cpp
)
