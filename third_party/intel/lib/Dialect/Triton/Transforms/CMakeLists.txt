add_triton_library(TritonIntelTransforms
  FoldTrueCmpIOp.cpp
  FuseReshape.cpp
  PrepareIfCombining.cpp
  RemoveMasks.cpp
  RewriteTensorDescriptorToPointer.cpp
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
