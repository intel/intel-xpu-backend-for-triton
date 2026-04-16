add_triton_library(TritonIntelTransforms
  FuseReshape.cpp
  RemoveMasks.cpp
  RewriteTensorDescriptorToPointer.cpp
  RewriteTensorPointer.cpp
  SimplifySignedArithmetic.cpp
  StrideVersioning.cpp

  DEPENDS
  TritonIntelTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIRTransformUtils
  TritonIntelAnalysis
  TritonIR
)
