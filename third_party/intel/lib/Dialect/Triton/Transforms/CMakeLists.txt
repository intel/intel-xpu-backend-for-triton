add_triton_library(TritonIntelTransforms
  FuseReshape.cpp
  RemoveBoundaryChecks.cpp
  RemoveMasks.cpp
  StrideVersioning.cpp
  BlockPointerToTensorDesc.cpp
  SimplifySignedArithmetic.cpp
  TensorDescToBlockPointer.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
