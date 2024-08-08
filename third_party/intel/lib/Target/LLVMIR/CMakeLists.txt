add_subdirectory(Dialect)

add_mlir_translation_library(PostProcessLLVMIR
  DSE.cpp
  LICM.cpp
  PostProcess.cpp
  SLPVectorizer.cpp
)
