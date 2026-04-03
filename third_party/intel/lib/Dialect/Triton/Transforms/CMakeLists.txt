add_triton_library(TritonIntelTransforms
  BlockPointerToTensorDesc.cpp
  FuseReshape.cpp
  RemoveMasks.cpp
  RewriteTensorDescriptorToPointer.cpp
  SimplifySignedArithmetic.cpp
  StrideVersioning.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIR
)
