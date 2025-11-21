add_triton_library(TritonIntelTransforms
  FuseReshape.cpp
  RemoveBoundaryChecks.cpp
  RemoveMasks.cpp
  StrideVersioning.cpp
  TensorDescToBlockPointer.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
