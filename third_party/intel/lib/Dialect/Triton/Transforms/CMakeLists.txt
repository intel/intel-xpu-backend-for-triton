add_triton_library(TritonIntelTransforms
  RemoveMasks.cpp
  TensorDescToBlockPointer.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
