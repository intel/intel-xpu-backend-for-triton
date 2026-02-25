add_triton_library(TritonIntelTransforms
  BlockPointerToTensorDesc.cpp
  FuseReshape.cpp
  FuseTranspose.cpp
  RemoveBoundaryChecks.cpp
  RemoveMasks.cpp
  RewriteTensorDescriptorToPointer.cpp
  SimplifySignedArithmetic.cpp
  StrideVersioning.cpp
  TensorDescToBlockPointer.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
